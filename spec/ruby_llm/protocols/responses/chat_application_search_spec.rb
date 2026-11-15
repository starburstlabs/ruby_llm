# frozen_string_literal: true

require 'spec_helper'
require_relative '../../../support/application_tool_search_helpers'

RSpec.describe RubyLLM::Protocols::Responses::Chat do
  include_context 'with application tool search'

  let(:protocol) { RubyLLM::Protocols::Responses.allocate }
  let(:tools) { { tool_search: finder, lookup_order: RubyLLM::Tool::Deferred.new(target) } }
  let(:model) { RubyLLM.models.find('gpt-5.4', provider: :openai) }
  let(:declaration) do
    { type: 'tool_search', execution: 'client', description: 'Finds tools for a task',
      parameters: finder.parameters_schema }
  end
  let(:lookup_definition) do
    { type: 'function', name: 'lookup_order', description: 'Looks up an order', parameters: target.parameters_schema,
      strict: false, defer_loading: true }
  end

  def search_call_item
    { type: 'tool_search_call', execution: 'client', call_id: 'search_1', status: 'completed',
      arguments: { 'query' => 'orders' } }
  end

  def render(messages = [], registered: tools, **options)
    protocol.send(:render_payload, messages, tools: registered, model:, temperature: nil, **options)
  end

  def request(messages = [], registered: tools, **options)
    protocol.send(:apply_tool_search, render(messages, registered:, **options), registered)
  end

  def parse(output)
    protocol.send(:parse_completion_body, responses_body(output), raw: nil)
  end

  def native_history(result = RubyLLM::ToolSearchResults.new(:lookup_order))
    [pending_search_message(raw: [client_search_item]), search_result_message(result)]
  end

  def search_output(call_id: 'search_1', tools: [lookup_definition])
    { type: 'tool_search_output', execution: 'client', call_id:, status: 'completed', tools: }
  end

  def types(items)
    items.map { |item| item[:type] || item['type'] }
  end

  def reasoning_item
    { 'type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'sealed' }
  end

  def raw_function_call(name: 'tool_search', call_id: 'search_1', arguments: '{"query":"orders"}', **item)
    { 'type' => 'function_call', 'id' => "fc_#{call_id}", 'call_id' => call_id, 'name' => name,
      'arguments' => arguments, 'status' => 'completed', **item }
  end

  def generated(chat, output)
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(json_response(responses_body(output)))
    chat.ask_later('Find order 42')
    chat.generate
  end

  it 'declares a nameless client search beside the deferred definitions, without the hosted search' do
    payload = request

    expect(payload[:tools]).to eq([declaration, lookup_definition])
    expect(protocol.send(:apply_tool_search, payload, tools)).to eq(payload)
  end

  it 'declares an ordinary function without a deferred tool' do
    declared = render(registered: { tool_search: finder, lookup_order: target })[:tools]

    expect(declared.map { |tool| tool.slice(:type, :name) })
      .to eq([{ type: 'function', name: 'tool_search' }, { type: 'function', name: 'lookup_order' }])
  end

  it 'renders a search call and its result as client items without raw content' do
    payload = render([pending_search_message, search_result_message])

    expect(payload[:input]).to eq([search_call_item, search_output])
  end

  it 'replays a raw ordinary call to the search tool as a client search, as it renders without raw content' do
    replayed = render([pending_search_message(raw: [reasoning_item, raw_function_call]), search_result_message])
    synthesized = render([pending_search_message, search_result_message])

    expect(replayed[:input]).to eq([reasoning_item, search_call_item, search_output])
    expect(replayed[:input].drop(1)).to eq(synthesized[:input])
  end

  it 'leaves the raw call to an ordinary tool in the same turn as it was' do
    order_call = raw_function_call(name: 'lookup_order', call_id: 'order_1', arguments: '{"order_id":"42"}')
    calls = [RubyLLM::ToolCall.new(id: 'search_1', name: 'tool_search', arguments: { 'query' => 'orders' }),
             RubyLLM::ToolCall.new(id: 'order_1', name: 'lookup_order', arguments: { 'order_id' => '42' })]
    message = RubyLLM::Message.new(role: :assistant, content: nil, tool_calls: calls.to_h { |call| [call.id, call] },
                                   raw_content: [reasoning_item, raw_function_call, order_call])
    result = RubyLLM::Message.new(role: :tool, content: 'Order 42 shipped', tool_call_id: 'order_1')

    expect(render([message, search_result_message, result])[:input])
      .to eq([reasoning_item, search_call_item, order_call, search_output,
              { type: 'function_call_output', call_id: 'order_1', output: 'Order 42 shipped' }])
  end

  it 'replays raw content that is no list of output items as it is' do
    raw = { 'response' => { 'id' => 'int_1', 'steps' => [] } }
    message = RubyLLM::Message.new(role: :assistant, content: 'Done', raw_content: raw)

    expect(render([message])[:input]).to eq([raw])
  end

  it 'answers a search replayed from the provider with the definitions it loaded' do
    payload = render(native_history)

    expect(types(payload[:input])).to eq(%w[tool_search_call tool_search_output])
    expect(payload[:input].last).to eq(search_output)
  end

  it 'parses the canonical search name and keeps the provider item for replay' do
    chat = RubyLLM::Chat.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)

    message = generated(chat, [client_search_item])

    expect(message.tool_calls.values).to contain_exactly(
      have_attributes(name: 'tool_search', namespace: nil, provider_call_id: 'search_1',
                      arguments: { 'query' => 'orders' })
    )
    expect(message.server_tool_calls).to eq([])
    expect(message.raw_content).to eq([client_search_item])
  end

  it 'keeps client search and ordinary calls distinct in the same provider output' do
    chat = RubyLLM::Chat.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true)
    order_call = raw_function_call(name: 'lookup_order', call_id: 'order_1', arguments: '{"order_id":"42"}')

    message = generated(chat, [client_search_item, order_call])
    result = RubyLLM::Message.new(role: :tool, content: 'Order 42 shipped', tool_call_id: 'order_1')

    expect(message.tool_calls.values).to contain_exactly(
      have_attributes(name: 'tool_search', provider_call_id: 'search_1', arguments: { 'query' => 'orders' }),
      have_attributes(name: 'lookup_order', provider_call_id: 'order_1', arguments: { 'order_id' => '42' })
    )
    expect(render([message, search_result_message, result])[:input])
      .to eq([client_search_item, order_call, search_output,
              { type: 'function_call_output', call_id: 'order_1', output: 'Order 42 shipped' }])
  end

  it 'leaves a hosted search call in the server history' do
    message = parse([client_search_item.merge('execution' => 'server', 'call_id' => nil)])

    expect(message.tool_calls).to be_empty
    expect(message.server_tool_calls.map(&:type)).to eq(['tool_search_call'])
  end

  it 'parses canonical search calls independently of tool registration' do
    message = parse([client_search_item])

    expect(message.tool_calls['search_1']).to have_attributes(name: 'tool_search', namespace: nil,
                                                              arguments: { 'query' => 'orders' })
    expect(types(render([message.without_native_content, search_result_message])[:input]))
      .to eq(%w[tool_search_call tool_search_output])
  end

  it 'parses malformed search arguments as an empty call' do
    call = parse([client_search_item(arguments: 'not json')]).tool_calls['search_1']
    message = RubyLLM::Message.new(role: :assistant, content: nil, tool_calls: { 'search_1' => call })

    expect(call.arguments).to eq({})
    expect(call).to be_malformed_arguments
    expect(render([message])[:input].first).to include(type: 'tool_search_call', arguments: {})
  end

  it 'raises for a search call whose ID another call shares' do
    duplicate = { 'type' => 'mcp_approval_request', 'id' => 'search_1', 'name' => 'lookup_order', 'arguments' => '{}' }
    output = [client_search_item, duplicate]

    expect { parse(output) }.to raise_error(RubyLLM::Error, /Duplicate tool call ID/)
    expect { protocol.send(:build_chunk, 'type' => 'response.completed', 'response' => responses_body(output)) }
      .to raise_error(RubyLLM::Error, /Duplicate tool call ID/)
  end

  it 'answers a result that loads nothing with an empty output' do
    expect(render(native_history(RubyLLM::ToolSearchResults.new))[:input].last).to eq(search_output(tools: []))
    expect(render(native_history(RubyLLM::ToolSearchResults.new(:retired)))[:input].last)
      .to eq(search_output(tools: []))
  end

  it 'drops a loaded name that is no longer registered, leaving the stored text as it is' do
    messages = native_history(RubyLLM::ToolSearchResults.new(:retired, :lookup_order))
    stored = messages.last.content.dup

    expect(render(messages)[:input].last).to eq(search_output)
    expect(messages.last.content).to eq(stored)
  end

  it 'sends attachments after the tool definitions returned by a client search' do
    image_url = 'https://example.com/order.png'
    image = RubyLLM::Attachment.new(image_url, mime_type: 'image/png')
    result = RubyLLM::Message.new(role: :tool, content: search_result_message.content, tool_call_id: 'search_1',
                                  attachments: [image])

    expect(render([pending_search_message(raw: [client_search_item]), result])[:input])
      .to eq([client_search_item, search_output,
              { role: 'user', content: [{ type: 'input_text', text: 'Attachments from tool call search_1:' },
                                        { type: 'input_image', image_url: }] }])
  end

  ['Denied by user', { error: 'search failed' }].each do |result|
    it "answers a result that is no list of tools with an empty output and its text, #{result.inspect}" do
      items = render(native_history(result))[:input]

      expect(items[-2]).to eq(search_output(tools: []))
      expect(items.last).to eq(role: 'user',
                               content: [{ type: 'input_text', text: 'Result of tool search search_1:' },
                                         { type: 'input_text', text: search_result_message(result).content }])
    end
  end

  it 'says when the search returned no output' do
    expect(render(native_history(nil))[:input].last[:content].last).to eq(type: 'input_text', text: '(no output)')
  end

  it 'answers the call a result follows, not every result' do
    messages = [*native_history, pending_search_message(name: 'lookup_order', call_id: 'order_1'),
                search_result_message(call_id: 'order_1')]
    input = render(messages)[:input]

    expect(types(input)).to eq(%w[tool_search_call tool_search_output function_call function_call_output])
    expect(input.last).to include(type: 'function_call_output', call_id: 'order_1',
                                  output: '{"loaded":["lookup_order"]}')
  end

  it 'keeps a hosted search of an earlier turn when a hosted search is declared beside the client search' do
    hosted = client_search_item.merge('execution' => 'server', 'call_id' => nil)
    payload = render([pending_search_message(raw: [hosted, client_search_item]), search_result_message])
    payload[:tools] << { type: 'tool_search' }

    expect(types(protocol.send(:apply_tool_search, payload, tools)[:input]))
      .to eq(%w[tool_search_call tool_search_call tool_search_output])
  end

  it 'strips both halves of a search once the search role is gone and nothing is deferred' do
    payload = request(native_history, registered: { lookup_order: target })

    expect(payload[:input]).to eq([])
    expect(payload[:tools]).to eq([lookup_definition.except(:defer_loading)])
  end

  it 'keeps both halves beside the hosted search once the search role is gone and a tool is deferred' do
    payload = request(native_history, registered: { lookup_order: tools[:lookup_order] })

    expect(types(payload[:input])).to eq(%w[tool_search_call tool_search_output])
    expect(payload[:tools]).to eq([lookup_definition, { type: 'tool_search' }])
  end

  it 'forces the search tool by type and other tools by name' do
    expect(render(tool_prefs: { choice: :tool_search })[:tool_choice]).to eq(type: 'tool_search')
    expect(render(tool_prefs: { choice: :lookup_order })[:tool_choice]).to eq(type: 'function', name: :lookup_order)
    expect(render(tool_prefs: { choice: :required })[:tool_choice]).to eq(:required)
  end

  it 'forces the search by type and refuses to force a namespaced target' do
    LookupOrder.namespace 'catalog'
    registered = { tool_search: finder, catalog_lookup_order: RubyLLM::Tool::Deferred.new(target) }

    expect(render(registered:, tool_prefs: { choice: :tool_search })[:tool_choice]).to eq(type: 'tool_search')
    expect { render(registered:, tool_prefs: { choice: :catalog_lookup_order }) }
      .to raise_error(RubyLLM::Error, 'The namespaced tool catalog_lookup_order cannot be forced; ' \
                                      'use choice: :required instead')
  end

  it 'counts tokens with the input and tools of the request' do
    messages = [pending_search_message, search_result_message]
    openai = RubyLLM::Providers::OpenAI::Responses.allocate
    rendered = openai.send(:render_payload, messages, tools:, model:, temperature: nil)
    requested = openai.send(:apply_tool_search, rendered, tools)
    counted = openai.send(:render_count_tokens_payload, messages, model:, tools:)

    expect(counted).to eq(requested.slice(*RubyLLM::Protocols::Responses::TokenCounting::COUNT_TOKENS_KEYS))
    expect(types(counted[:input])).to eq(%w[tool_search_call tool_search_output])
  end

  it 'renders repeatedly without changing the messages' do
    messages = [pending_search_message, search_result_message]
    original = messages.map(&:to_h)
    first = render(messages)

    expect(render(messages)).to eq(first)
    expect(messages.map(&:to_h)).to eq(original)
  end

  it 'parses a client search call only in the terminal stream event' do
    accumulator = RubyLLM::Protocol::StreamAccumulator.new
    added = protocol.send(:build_chunk, 'type' => 'response.output_item.added', 'output_index' => 0,
                                        'item' => client_search_item)
    done = protocol.send(:build_chunk, 'type' => 'response.output_item.done', 'output_index' => 0,
                                       'item' => client_search_item)
    final = protocol.send(:build_chunk, 'type' => 'response.completed',
                                        'response' => responses_body([client_search_item]))
    [added, done, final].each { |chunk| accumulator.add(chunk) }
    streamed = accumulator.to_message(nil)
    sync = parse([client_search_item])

    expect(added.tool_calls).to be_empty
    expect(done.tool_calls).to be_empty
    expect(streamed.tool_calls.map(&:to_h)).to eq(sync.tool_calls.map(&:to_h))
    expect(streamed.tool_calls.map(&:name)).to eq(['tool_search'])
    expect(streamed.raw_content).to eq(sync.raw_content)
  end
end
