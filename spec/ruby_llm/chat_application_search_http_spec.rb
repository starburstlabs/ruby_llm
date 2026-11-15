# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/application_tool_search_helpers'

RSpec.describe RubyLLM::Chat do
  include_context 'with application tool search'

  def fixture(provider, step)
    JSON.parse(File.read(File.expand_path("../fixtures/tool_search/#{provider}_#{step}.json", __dir__)))
  end

  def event_stream(events)
    { status: 200, headers: { 'Content-Type' => 'text/event-stream' },
      body: events.map { |event| "data: #{JSON.generate(event)}\n\n" }.join }
  end

  def response_stream(body)
    item = body.fetch('output').first
    events = [{ 'type' => 'response.output_item.added', 'output_index' => 0, 'item' => item }]
    if item['type'] == 'function_call'
      events << { 'type' => 'response.function_call_arguments.delta', 'output_index' => 0,
                  'delta' => item['arguments'] }
    elsif item['type'] == 'message'
      events << { 'type' => 'response.output_text.delta', 'output_index' => 0,
                  'delta' => item['content'].first['text'] }
    end
    events << { 'type' => 'response.output_item.done', 'output_index' => 0, 'item' => item }
    events << { 'type' => 'response.completed', 'response' => body }
    event_stream(events)
  end

  def anthropic_stream(body)
    block = body.fetch('content').first
    start = block['type'] == 'tool_use' ? block.merge('input' => {}) : block.merge('text' => '')
    delta = if block['type'] == 'tool_use'
              { 'type' => 'input_json_delta', 'partial_json' => JSON.generate(block['input']) }
            else
              { 'type' => 'text_delta', 'text' => block['text'] }
            end
    event_stream([
                   { 'type' => 'message_start', 'message' => body.merge('content' => [], 'stop_reason' => nil) },
                   { 'type' => 'content_block_start', 'index' => 0, 'content_block' => start },
                   { 'type' => 'content_block_delta', 'index' => 0, 'delta' => delta },
                   { 'type' => 'content_block_stop', 'index' => 0 },
                   { 'type' => 'message_delta', 'delta' => { 'stop_reason' => body['stop_reason'] },
                     'usage' => { 'output_tokens' => 3 } },
                   { 'type' => 'message_stop' }
                 ])
  end

  def chat_on(provider, model)
    described_class.new(model:, provider:).with_tools(finder, target, defer: true)
  end

  def stub_replies(url, replies)
    requests = []
    stub_request(:post, url).to_return do |request|
      requests << JSON.parse(request.body)
      replies.shift || raise("Unexpected request to #{url} after scripted replies were exhausted")
    end
    requests
  end

  def types(items)
    items.map { |item| item[:type] || item['type'] }
  end

  [false, true].each do |stream|
    context "with streaming #{stream}" do
      it 'searches, loads, and executes a target through Responses' do
        chat = chat_on(:openai, 'gpt-5.4')
        replies = %w[search target answer].map do |step|
          body = fixture(:responses, step)
          stream ? response_stream(body) : json_response(body)
        end
        requests = stub_replies('https://api.openai.com/v1/responses', replies)

        reply = stream ? chat.ask('Find order 42') { |_chunk| nil } : chat.ask('Find order 42')

        expect(reply.content).to eq('Order 42 shipped')
        expect(finder.queries).to eq(['orders'])
        expect(requests.size).to eq(3)
        requests.each do |request|
          expect(request.fetch('tools')).to match([hash_including('name' => 'lookup_order', 'defer_loading' => true),
                                                   hash_including('type' => 'tool_search', 'execution' => 'client')])
        end
        search_output = requests[1]['input'].find { |item| item['type'] == 'tool_search_output' }
        expect(search_output).to include('execution' => 'client', 'call_id' => 'search_1', 'status' => 'completed')
        loaded = search_output.fetch('tools')
        expect(loaded.map { |tool| tool.slice('type', 'name', 'defer_loading') })
          .to eq([{ 'type' => 'function', 'name' => 'lookup_order', 'defer_loading' => true }])
        expect(loaded.first.dig('parameters', 'required')).to eq(['order_id'])
        expect(requests[2]['input']).to include(hash_including('type' => 'function_call', 'call_id' => 'order_1',
                                                               'namespace' => 'lookup_order'))
        expect(requests[2]['input'].last).to include('type' => 'function_call_output', 'call_id' => 'order_1',
                                                     'output' => 'Order 42 shipped')
        expect(chat.messages.select(&:tool_result?).map(&:provider_call_id)).to eq(%w[search_1 order_1])
        expect(chat.deferred_tools.keys).to eq([:lookup_order])
      end

      it 'searches, loads, and executes a target through Anthropic' do
        chat = chat_on(:anthropic, model_for(:anthropic))
        replies = %w[search target answer].map do |step|
          body = fixture(:anthropic, step)
          stream ? anthropic_stream(body) : json_response(body)
        end
        requests = stub_replies('https://api.anthropic.com/v1/messages', replies)

        reply = stream ? chat.ask('Find order 42') { |_chunk| nil } : chat.ask('Find order 42')

        expect(reply.content).to eq('Order 42 shipped')
        expect(finder.queries).to eq(['orders'])
        expect(requests.size).to eq(3)
        expect(requests[0]['tools'].map { |tool| tool.slice('name', 'defer_loading') })
          .to eq([{ 'name' => 'lookup_order', 'defer_loading' => true }, { 'name' => 'tool_search' }])
        expect(requests[1]['messages'].last['content'].first).to eq(
          'type' => 'tool_result', 'tool_use_id' => 'search_1',
          'content' => [{ 'type' => 'tool_reference', 'tool_name' => 'lookup_order' }]
        )
        expect(requests[2]['messages'].last['content'].first['content']).to eq(
          [{ 'type' => 'text', 'text' => 'Order 42 shipped' }]
        )
        expect(chat.deferred_tools.keys).to eq([:lookup_order])
      end
    end
  end

  it 'answers a declined search with an empty output and the text of the refusal' do
    FindTools.requires_approval
    chat = chat_on(:openai, 'gpt-5.4')
    stub_replies('https://api.openai.com/v1/responses', [json_response(fixture(:responses, :search))])
    chat.ask('Find order 42')
    chat.decline(chat.pending_questions.find(&:approval?))
    chat.run_tools
    input = chat.render[:input]

    expect(finder.queries).to eq([])
    expect(input[-2]).to include(type: 'tool_search_output', call_id: 'search_1', tools: [])
    expect(input.last[:content].last[:text]).to match(/declined|denied/i)
  end

  it 'answers a search the model called with the wrong arguments with an empty output and the error' do
    chat = chat_on(:openai, 'gpt-5.4')
    search = fixture(:responses, :search)
    search['output'].first['arguments'] = {}
    requests = stub_replies('https://api.openai.com/v1/responses',
                            [json_response(search), json_response(fixture(:responses, :answer))])

    chat.ask('Find order 42')

    expect(finder.queries).to eq([])
    expect(requests.last['input'][-2]).to include('type' => 'tool_search_output', 'call_id' => 'search_1',
                                                  'tools' => [])
    expect(requests.last['input'].last['content'].last['text']).to include('query')
  end

  it 'keeps the provider items on a same-provider model switch' do
    chat = chat_on(:openai, 'gpt-5.4')
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(fixture(:responses, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools
    chat.with_model('gpt-5.4-mini')
    input = chat.render[:input]

    expect(input[1]).to include('type' => 'tool_search_call', 'id' => 'item_search', 'call_id' => 'search_1')
    expect(input.last).to include(type: 'tool_search_output', call_id: 'search_1')
  end

  it 'renders Responses history as an ordinary call with references after switching to Anthropic' do
    chat = chat_on(:openai, 'gpt-5.4')
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(fixture(:responses, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools
    chat.with_model(model_for(:anthropic), provider: :anthropic)
    payload = chat.render

    expect(payload[:messages][1][:content].first).to include(type: 'tool_use', id: 'search_1', name: 'tool_search')
    expect(payload[:messages].last[:content].first[:content]).to eq([{ type: 'tool_reference',
                                                                       tool_name: 'lookup_order' }])
    expect(payload.to_json).not_to include('tool_search_call')
  end

  it 'renders Anthropic history as client search items after switching to Responses' do
    chat = chat_on(:anthropic, model_for(:anthropic))
    stub_request(:post, 'https://api.anthropic.com/v1/messages').to_return(json_response(fixture(:anthropic, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools
    chat.with_model('gpt-5.4', provider: :openai)
    input = chat.render[:input]

    expect(input[1]).to eq(type: 'tool_search_call', execution: 'client', call_id: 'search_1', status: 'completed',
                           arguments: { 'query' => 'orders' })
    expect(input.last).to include(type: 'tool_search_output', execution: 'client', call_id: 'search_1')
    expect(input.last[:tools].map { |tool| tool[:name] }).to eq(['lookup_order'])
  end

  it 'replays one history natively, as text on a provider without tool search, and natively again' do
    chat = chat_on(:openai, 'gpt-5.4')
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(fixture(:responses, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools

    chat.with_model(model_for(:gemini), provider: :gemini)
    gemini = chat.render
    declared = gemini[:tools].flat_map { |tool| tool[:functionDeclarations] }.map { |declaration| declaration[:name] }

    expect(declared).to eq(%w[tool_search lookup_order])
    expect(gemini.to_json).not_to include('tool_search_call', 'defer_loading')

    chat.with_model('gpt-5.4', provider: :openai)

    expect(types(chat.render[:input]).last(2)).to eq(%w[tool_search_call tool_search_output])
  end
end
