# frozen_string_literal: true

require 'spec_helper'
require_relative '../../../support/client_tool_search_helpers'

RSpec.describe RubyLLM::Protocols::Responses::Chat do
  include_context 'with client tool search'

  let(:protocol) do
    RubyLLM::Protocols::Responses.allocate
  end
  let(:tools) { { find_tools: finder, lookup_order: RubyLLM::Tool::Deferred.new(target) } }
  let(:model) { RubyLLM.models.find('gpt-5.4', provider: :openai) }

  def render(messages = [], registered: tools, **options)
    protocol.send(:render_payload, messages, tools: registered, model:, temperature: nil, **options)
  end

  def native_history(result = RubyLLM::ToolSearchResults.new(:lookup_order))
    [pending_search_message(raw: [client_search_item]), search_result_message(result)]
  end

  it 'declares a nameless client search and preserves all target definitions' do
    declarations = render[:tools]

    expect(declarations.first).to eq(type: 'tool_search', execution: 'client', description: finder.description,
                                     parameters: finder.parameters_schema)
    expect(declarations.last).to eq(protocol.send(:tool_for, tools[:lookup_order]))
    expect(protocol.send(:apply_tool_search, render)[:tools]).to eq(declarations)
  end

  it 'normalizes client calls by call_id and the real Ruby handler name' do
    render
    message = protocol.send(:parse_completion_body, responses_body([client_search_item]), raw: nil)

    expect(message.tool_calls.keys).to eq(['search_1'])
    expect(message.tool_calls['search_1']).to have_attributes(name: 'find_tools', arguments: { 'query' => 'orders' })
    expect(message.server_tool_calls).to eq([])
    expect(message.raw_content).to eq([client_search_item])
  end

  it 'leaves hosted calls in server history' do
    render
    hosted = client_search_item.merge('execution' => 'server', 'call_id' => nil)
    message = protocol.send(:parse_completion_body, responses_body([hosted]), raw: nil)

    expect(message.tool_calls).to be_nil
    expect(message.server_tool_calls.first.type).to eq('tool_search_call')
  end

  it 'raises a parse error if the handler is absent' do
    render(registered: {})

    expect do
      protocol.send(:parse_completion_body, responses_body([client_search_item]), raw: nil)
    end.to raise_error(RubyLLM::ToolCallParseError, /searches_tools/)
  end

  it 'requires object arguments for client search calls' do
    render

    expect do
      protocol.send(:parse_completion_body, responses_body([client_search_item(arguments: '{"query":"orders"}')]),
                    raw: nil)
    end.to raise_error(RubyLLM::ToolCallParseError)
  end

  [nil, ''].each do |call_id|
    it "rejects a missing native call mapping, #{call_id.inspect}" do
      render
      expect do
        protocol.send(:parse_completion_body, responses_body([client_search_item(call_id:)]), raw: nil)
      end.to raise_error(RubyLLM::ToolCallParseError)
    end
  end

  %w[tool_search_call function_call mcp_approval_request].each do |type|
    it "rejects a native ID colliding with another #{type}" do
      render
      duplicate = client_search_item.merge('type' => type, 'id' => 'search_1',
                                           'name' => 'lookup_order', 'arguments' => '{}')
      duplicate['arguments'] = {} if type == 'tool_search_call'
      output = [client_search_item, duplicate]

      expect do
        protocol.send(:parse_completion_body, responses_body(output), raw: nil)
      end.to raise_error(RubyLLM::ToolCallParseError)
      expect do
        protocol.send(:build_chunk, 'type' => 'response.completed', 'response' => responses_body(output))
      end.to raise_error(RubyLLM::ToolCallParseError)
    end
  end

  it 'renders exact tool definitions including deferral flags in native outputs' do
    payload = render(native_history)

    expect(payload[:input].last).to eq(type: 'tool_search_output', execution: 'client', call_id: 'search_1',
                                       status: 'completed', tools: [payload[:tools].last])
  end

  it 'accepts eager matches without promoting deferred definitions' do
    registered = tools.merge(lookup_order: target)
    payload = render(native_history, registered:)

    expect(payload[:input].last[:tools]).to eq([payload[:tools].last])
    expect(tools[:lookup_order]).to be_a(RubyLLM::Tool::Deferred)
  end

  it 'maps outputs by the replayed native call ID rather than the current handler name or envelope' do
    messages = native_history + [pending_search_message(call_id: 'ordinary'),
                                 search_result_message(call_id: 'ordinary')]
    output = render(messages)[:input]

    expect(output[1][:type]).to eq('tool_search_output')
    expect(output.last).to include(type: 'function_call_output', call_id: 'ordinary')
  end

  it 'keeps native history when the search handler is renamed' do
    allow(finder).to receive(:name).and_return('new_search')

    expect(render(native_history)[:input].last[:type]).to eq('tool_search_output')
  end

  [false, true].each do |deferred|
    it "downgrades both sides after removing the search role with deferred targets #{deferred}" do
      registered = { lookup_order: deferred ? tools[:lookup_order] : target }
      payload = protocol.send(:apply_tool_search, render(native_history, registered:))

      expect(payload[:input].first).to eq(type: 'function_call', name: 'find_tools', call_id: 'search_1',
                                          arguments: '{"query":"orders"}')
      expect(payload[:input].last[:type]).to eq('function_call_output')
    end
  end

  it 'drops reasoning tied to a client call when the search role is removed' do
    reasoning = { 'type' => 'reasoning', 'id' => 'rs_1', 'encrypted_content' => 'opaque', 'summary' => [] }
    raw = [reasoning, client_search_item]
    messages = [pending_search_message(raw:), search_result_message]

    expect(render(messages)[:input].first).to eq(reasoning)

    input = render(messages, registered: { lookup_order: tools[:lookup_order] })[:input]

    expect(input.map { |item| item[:type] || item['type'] }).to eq(%w[function_call function_call_output])
    expect(messages.first.raw_content).to eq(raw)
  end

  it 'warns and omits retired results while keeping the stored envelope unchanged' do
    messages = native_history(RubyLLM::ToolSearchResults.new(:retired, :lookup_order))
    content = messages.last.content.dup
    allow(RubyLLM.logger).to receive(:warn)

    result = render(messages)[:input].last

    expect(result[:tools].map { |tool| tool[:name] }).to eq(['lookup_order'])
    expect(RubyLLM.logger).to have_received(:warn).with(/retired/)
    expect(messages.last.content).to eq(content)
  end

  it 'renders empty results as an empty native output' do
    expect(render(native_history(RubyLLM::ToolSearchResults.new))[:input].last)
      .to include(type: 'tool_search_output', tools: [])
  end

  ['Denied by user', { error: 'search failed' }, 'Unexpected result'].each do |result|
    it "keeps the native output and adds diagnostic text for #{result.inspect}" do
      items = render(native_history(result))[:input]

      expect(items[-2]).to include(type: 'tool_search_output', tools: [], status: 'completed')
      expect(items.last).to eq(role: 'user',
                               content: [{ type: 'input_text',
                                           text: search_result_message(result).content }])
    end
  end

  it 'explains a search handler that returns no output' do
    items = render(native_history(nil))[:input]

    expect(items[-2]).to include(type: 'tool_search_output', tools: [])
    expect(items.last).to eq(role: 'user', content: [{ type: 'input_text', text: '(no output)' }])
  end

  it 'strips obsolete hosted search items under a client-only declaration' do
    hosted = client_search_item.merge('execution' => 'server', 'call_id' => nil)
    raw = [hosted, { 'type' => 'tool_search_output', 'tools' => [] }, client_search_item]
    messages = [pending_search_message(raw:), search_result_message]
    payload = protocol.send(:apply_tool_search, render(messages))

    expect(payload[:input].map { |item| item[:type] || item['type'] }).to eq(%w[tool_search_call tool_search_output])
    expect(messages.first.raw_content).to eq(raw)
  end

  it 'preserves hosted history when a hosted search is explicitly declared beside the client search' do
    hosted = client_search_item.merge('execution' => 'server', 'call_id' => nil)
    messages = [pending_search_message(raw: [hosted, client_search_item]), search_result_message]
    payload = render(messages)
    payload[:tools] << { type: 'tool_search' }

    expect(protocol.send(:apply_tool_search, payload)[:input].first).to eq(hosted)
  end

  it 'renders repeatedly without changing messages or definitions' do
    messages = native_history
    original = messages.map(&:to_h)
    first = render(messages)

    expect(render(messages)).to eq(first)
    expect(messages.map(&:to_h)).to eq(original)
  end

  it 'does not retain a call ID that a later render no longer replays' do
    render(native_history)

    expect(render([search_result_message])[:input].last[:type]).to eq('function_call_output')
  end

  it 'rejects a named forced search choice while preserving ordinary choices' do
    expect { render(tool_prefs: { choice: :find_tools }) }.to raise_error(ArgumentError, /cannot force a named/)
    expect(render(tool_prefs: { choice: :lookup_order })[:tool_choice]).to eq(type: 'function', name: :lookup_order)
    expect(render(tool_prefs: { choice: :required })[:tool_choice]).to eq(:required)
  end

  it 'normalizes a complete search call only in the terminal stream event' do
    render
    accumulator = RubyLLM::Protocol::StreamAccumulator.new
    added = protocol.send(:build_chunk, 'type' => 'response.output_item.added', 'output_index' => 0,
                                        'item' => client_search_item)
    done = protocol.send(:build_chunk, 'type' => 'response.output_item.done', 'output_index' => 0,
                                       'item' => client_search_item)
    final = protocol.send(:build_chunk, 'type' => 'response.completed',
                                        'response' => responses_body([client_search_item]))
    [added, done, final].each { |chunk| accumulator.add(chunk) }
    streamed = accumulator.to_message(nil)
    sync = protocol.send(:parse_completion_body, responses_body([client_search_item]), raw: nil)

    expect(added.tool_calls).to be_nil
    expect(done.tool_calls).to be_nil
    expect(streamed.tool_calls.transform_values(&:to_h)).to eq(sync.tool_calls.transform_values(&:to_h))
    expect(streamed.raw_content).to eq(sync.raw_content)
  end
end
