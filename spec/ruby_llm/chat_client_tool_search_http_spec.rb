# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/client_tool_search_helpers'

RSpec.describe RubyLLM::Chat do
  include_context 'with client tool search'

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

  [false, true].each do |stream|
    context "with streaming #{stream}" do
      it 'searches, loads, and executes a target through Responses HTTP' do
        chat = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)
        requests = []
        replies = %w[search target answer].map do |step|
          body = fixture(:responses, step)
          stream ? response_stream(body) : json_response(body)
        end
        stub_request(:post, 'https://api.openai.com/v1/responses').to_return do |request|
          requests << JSON.parse(request.body)
          replies.shift
        end

        reply = stream ? chat.ask('Find order 42') { |_chunk| nil } : chat.ask('Find order 42')

        expect(reply.content).to eq('Order 42 shipped')
        expect(finder.queries).to eq(['orders'])
        expect(requests.size).to eq(3)
        search_output = requests[1]['input'].find { |item| item['type'] == 'tool_search_output' }
        expect(search_output).to eq('type' => 'tool_search_output', 'execution' => 'client', 'call_id' => 'search_1',
                                    'status' => 'completed', 'tools' => [requests[1]['tools'].last])
        expect(requests[2]['input'].last).to include('type' => 'function_call_output', 'call_id' => 'order_1',
                                                     'output' => 'Order 42 shipped')
        expect(chat.messages.select(&:tool_result?).map(&:tool_call_id)).to eq(%w[search_1 order_1])
        expect(chat.deferred_tools.keys).to eq([:lookup_order])
      end

      it 'searches, loads, and executes a target through Anthropic HTTP' do
        chat = described_class.new(model: model_for(:anthropic), provider: :anthropic)
                              .with_tools(finder, target, defer: true)
        requests = []
        replies = %w[search target answer].map do |step|
          body = fixture(:anthropic, step)
          stream ? anthropic_stream(body) : json_response(body)
        end
        stub_request(:post, 'https://api.anthropic.com/v1/messages').to_return do |request|
          requests << JSON.parse(request.body)
          replies.shift
        end

        reply = stream ? chat.ask('Find order 42') { |_chunk| nil } : chat.ask('Find order 42')

        expect(reply.content).to eq('Order 42 shipped')
        expect(finder.queries).to eq(['orders'])
        expect(requests.size).to eq(3)
        expect(requests[1]['messages'].last['content'].first).to eq(
          'type' => 'tool_result', 'tool_use_id' => 'search_1',
          'content' => [{ 'type' => 'tool_reference', 'tool_name' => 'lookup_order' }]
        )
        expect(requests[0]['tools'].map { |tool| tool['name'] }).to eq(%w[find_tools lookup_order])
        expect(requests[2]['messages'].last['content'].first['content']).to eq(
          [{ 'type' => 'text', 'text' => 'Order 42 shipped' }]
        )
        expect(chat.deferred_tools.keys).to eq([:lookup_order])
      end
    end
  end

  it 'uses a native empty output and diagnostic text after denial' do
    FindTools.requires_approval
    chat = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(fixture(:responses, :search)))
    chat.ask('Find order 42')
    expect(chat.pending_approvals.map(&:id)).to eq(['search_1'])

    chat.deny('search_1')
    chat.run_tools
    input = chat.render[:input]

    expect(finder.queries).to eq([])
    expect(input[-2]).to include(type: 'tool_search_output', call_id: 'search_1', tools: [])
    expect(input.last[:content].first[:text]).to match(/denied/i)
  end

  it 'answers invalid search arguments with a native empty output and the argument error' do
    chat = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)
    search = fixture(:responses, :search)
    search['output'].first['arguments'] = {}
    requests = []
    replies = [json_response(search), json_response(fixture(:responses, :answer))]
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return do |request|
      requests << JSON.parse(request.body)
      replies.shift
    end

    chat.ask('Find order 42')

    expect(finder.queries).to eq([])
    expect(requests.last['input'][-2]).to include('type' => 'tool_search_output', 'call_id' => 'search_1',
                                                  'tools' => [])
    expect(requests.last['input'].last['content'].first['text']).to include('missing keyword: query')
  end

  it 'keeps native history on a same-provider model switch' do
    chat = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(fixture(:responses, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools
    chat.with_model(model_for(:openai), provider: :openai, protocol: :responses)

    expect(chat.render[:input].last[:type]).to eq('tool_search_output')
  end

  it 'renders Responses history as ordinary calls with references after switching to Anthropic' do
    chat = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(fixture(:responses, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools
    chat.with_model(model_for(:anthropic), provider: :anthropic)
    payload = chat.render

    expect(payload[:messages][1][:content].first).to include(type: 'tool_use', id: 'search_1', name: 'find_tools')
    expect(payload[:messages].last[:content].first[:content]).to eq([{ type: 'tool_reference',
                                                                       tool_name: 'lookup_order' }])
    expect(payload.to_json).not_to include('tool_search_call')
  end

  it 'keeps Anthropic-origin pairs ordinary after switching to Responses' do
    chat = described_class.new(model: model_for(:anthropic), provider: :anthropic).with_tools(finder, target,
                                                                                              defer: true)
    stub_request(:post, 'https://api.anthropic.com/v1/messages').to_return(json_response(fixture(:anthropic, :search)))
    chat.ask_later('Find order 42')
    chat.generate
    chat.run_tools
    chat.with_model('gpt-5.4', provider: :openai)
    input = chat.render[:input]

    expect(input[1]).to include(type: 'function_call', call_id: 'search_1', name: 'find_tools')
    expect(input.last).to include(type: 'function_call_output', call_id: 'search_1')
    expect(input.last[:output]).to eq('{"tool_search_results":["lookup_order"]}')
  end
end
