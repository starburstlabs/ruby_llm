# frozen_string_literal: true

require 'spec_helper'
require_relative '../../../support/application_tool_search_helpers'

RSpec.describe RubyLLM::Protocols::Anthropic::Chat do
  include_context 'with application tool search'

  let(:protocol) { RubyLLM::Protocols::Anthropic.allocate }
  let(:tools) { { tool_search: finder, lookup_order: RubyLLM::Tool::Deferred.new(target) } }
  let(:model) { RubyLLM.models.find(model_for(:anthropic), provider: :anthropic) }
  let(:reference) { [{ type: 'tool_reference', tool_name: 'lookup_order' }] }

  def render(messages = [], registered: tools, **options)
    protocol.send(:render_payload, messages, tools: registered, model:, temperature: nil, **options)
  end

  def request(messages = [], registered: tools, **options)
    protocol.send(:apply_tool_search, render(messages, registered:, **options), registered)
  end

  def result_content(payload)
    payload[:messages].last[:content].first[:content]
  end

  it 'declares the search tool as an eager function beside the deferred tools, without the hosted search' do
    expect(request[:tools]).to eq([
                                    { name: 'tool_search', description: 'Finds tools for a task',
                                      input_schema: finder.parameters_schema },
                                    { name: 'lookup_order', description: 'Looks up an order',
                                      input_schema: target.parameters_schema, defer_loading: true }
                                  ])
  end

  it 'adds the hosted search when the request defers tools without a search tool of its own' do
    declared = request(registered: tools.except(:tool_search))[:tools]

    expect(declared.last).to eq(RubyLLM::Protocols::Anthropic::Tools::NATIVE_TOOL_SEARCH)
  end

  it 'renders what a search loaded as references to the deferred tools' do
    payload = render([pending_search_message, search_result_message])

    expect(payload[:messages].last[:content].first)
      .to eq(type: 'tool_result', tool_use_id: 'search_1', content: reference)
  end

  it 'references a namespaced deferred tool by the name the request declares' do
    lookup = Class.new(RubyLLM::Tool) do
      namespace 'crm'
      description 'Finds a customer'
      def self.tool_name = 'customer_lookup'
      def execute = 'Ada'
    end
    flattened = RubyLLM::Tool::Flattened.wrap(RubyLLM::Tool::Deferred.new(lookup.new))
    messages = [pending_search_message, search_result_message(RubyLLM::ToolSearchResults.new(lookup))]

    expect(result_content(render(messages, registered: tools.merge(crm_customer_lookup: flattened))))
      .to eq([{ type: 'tool_reference', tool_name: 'crm_customer_lookup' }])
  end

  it 'drops a loaded name that is no longer registered, leaving the stored text as it is' do
    retired = search_result_message(RubyLLM::ToolSearchResults.new(:retired, :lookup_order))
    stored = retired.content.dup

    expect(result_content(render([pending_search_message, retired]))).to eq(reference)
    expect(retired.content).to eq(stored)
  end

  it 'references deferred tools only' do
    registered = tools.merge(lookup_order: target, weather: RubyLLM::Tool::Deferred.new(RubyLLM::Tool.new))

    expect(result_content(render([pending_search_message, search_result_message], registered:)))
      .to eq([{ type: 'text', text: '{"loaded":["lookup_order"]}' }])
  end

  it 'sends a result that loads nothing, and an error, as the text they stored' do
    nothing = search_result_message(RubyLLM::ToolSearchResults.new(:retired))
    error = search_result_message({ error: 'search failed' })

    expect(result_content(render([pending_search_message, nothing])))
      .to eq([{ type: 'text', text: '{"loaded":["retired"]}' }])
    expect(result_content(render([pending_search_message, error])))
      .to eq([{ type: 'text', text: '{"error":"search failed"}' }])
  end

  it 'references what the built-in tool_search found before the chat registered its own search' do
    messages = [pending_search_message(name: 'tool_search'),
                search_result_message({ loaded: %w[lookup_order], tools: [] })]

    expect(result_content(render(messages))).to eq(reference)
  end

  it 'leaves the result of a tool that is not the search as text, whatever it contains' do
    messages = [pending_search_message(name: 'lookup_order'), search_result_message]

    expect(result_content(render(messages))).to eq([{ type: 'text', text: '{"loaded":["lookup_order"]}' }])
  end

  it 'changes nothing without a deferred tool' do
    registered = { tool_search: finder, lookup_order: target }
    payload = request([pending_search_message, search_result_message], registered:)

    expect(payload[:tools].map { |tool| tool[:name] }).to eq(%w[tool_search lookup_order])
    expect(result_content(payload)).to eq([{ type: 'text', text: '{"loaded":["lookup_order"]}' }])
  end

  it 'strips hosted search blocks while keeping ordinary tool calls' do
    raw = [
      { 'type' => 'server_tool_use', 'id' => 'hosted', 'name' => 'tool_search_tool_bm25', 'input' => {} },
      { 'type' => 'tool_search_tool_result', 'tool_use_id' => 'hosted', 'content' => [] },
      { 'type' => 'tool_use', 'id' => 'search_1', 'name' => 'tool_search', 'input' => { 'query' => 'orders' } }
    ]
    messages = [pending_search_message(raw:), search_result_message]
    payload = request(messages)

    expect(payload[:messages].first[:content]).to eq([raw.last])
    expect(messages.first.raw_content).to eq(raw)
  end

  it 'counts tokens with the tools and references of the request' do
    messages = [pending_search_message, search_result_message]
    counted = protocol.send(:render_count_tokens_payload, messages, tools:, model:)

    expect(counted).to eq(request(messages).slice(*described_class::COUNT_TOKENS_KEYS))
    expect(result_content(counted)).to eq(reference)
  end

  it 'forces the search tool by name' do
    expect(render(tool_prefs: { choice: :tool_search })[:tool_choice]).to eq(type: :tool, name: :tool_search)
  end

  it 'renders the same payload repeatedly without changing the messages' do
    messages = [pending_search_message, search_result_message]
    original = messages.map(&:to_h)
    first = render(messages)

    expect(render(messages)).to eq(first)
    expect(messages.map(&:to_h)).to eq(original)
  end
end
