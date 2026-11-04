# frozen_string_literal: true

require 'spec_helper'
require_relative '../../../support/client_tool_search_helpers'

RSpec.describe RubyLLM::Protocols::Anthropic::Chat do
  include_context 'with client tool search'

  let(:protocol) { RubyLLM::Protocols::Anthropic.allocate }
  let(:tools) { { find_tools: finder, lookup_order: RubyLLM::Tool::Deferred.new(target) } }
  let(:model) { RubyLLM.models.find(model_for(:anthropic), provider: :anthropic) }

  def render(messages = [], **options)
    protocol.send(:render_payload, messages, tools:, model:, temperature: nil, **options)
  end

  it 'declares the search handler as an eager ordinary function without a hosted search' do
    payload = protocol.send(:apply_tool_search, render)

    expect(payload[:tools].first).to eq(name: 'find_tools', description: finder.description,
                                        input_schema: finder.parameters_schema)
    expect(payload[:tools].last).to include(name: 'lookup_order', defer_loading: true)
    expect(payload[:tools].size).to eq(2)
  end

  it 'renders neutral results as references to current registered definitions' do
    payload = render([pending_search_message, search_result_message])

    expect(payload[:messages].last[:content].first).to eq(
      type: 'tool_result', tool_use_id: 'search_1',
      content: [{ type: 'tool_reference', tool_name: 'lookup_order' }]
    )
  end

  it 'uses no-match text for an empty result' do
    payload = render([search_result_message(RubyLLM::ToolSearchResults.new)])

    expect(payload[:messages].last[:content].first[:content]).to eq([{ type: 'text',
                                                                       text: 'No matching tools found.' }])
  end

  it 'leaves ordinary tool errors unchanged' do
    payload = render([search_result_message({ error: 'search failed' })])

    expect(payload[:messages].last[:content].first[:content]).to eq([{ type: 'text',
                                                                       text: '{"error":"search failed"}' }])
  end

  it 'omits retired targets with a warning without modifying stored content' do
    message = search_result_message(RubyLLM::ToolSearchResults.new(:retired, :lookup_order))
    original = message.content.dup
    allow(RubyLLM.logger).to receive(:warn)

    payload = render([message])

    expect(payload[:messages].last[:content].first[:content]).to eq([{ type: 'tool_reference',
                                                                       tool_name: 'lookup_order' }])
    expect(RubyLLM.logger).to have_received(:warn).with(/retired/)
    expect(message.content).to eq(original)
  end

  it 'strips hosted search blocks while retaining ordinary tool calls' do
    raw = [
      { 'type' => 'server_tool_use', 'id' => 'hosted', 'name' => 'tool_search_tool_bm25', 'input' => {} },
      { 'type' => 'tool_search_tool_result', 'tool_use_id' => 'hosted', 'content' => [] },
      { 'type' => 'tool_use', 'id' => 'search_1', 'name' => 'find_tools', 'input' => { 'query' => 'orders' } }
    ]
    messages = [pending_search_message(raw:), search_result_message]
    payload = protocol.send(:apply_tool_search, render(messages))

    expect(payload[:messages].first[:content]).to eq([raw.last])
    expect(messages.first.raw_content).to eq(raw)
  end

  it 'uses identical tools and references when counting tokens' do
    messages = [pending_search_message, search_result_message]
    normal = protocol.send(:apply_tool_search, render(messages))
    counted = protocol.send(:render_count_tokens_payload, messages, tools:, model:)

    expect(counted).to eq(normal.slice(*described_class::COUNT_TOKENS_KEYS))
  end

  it 'keeps named forced search choices' do
    expect(render(tool_prefs: { choice: :find_tools })[:tool_choice]).to eq(type: :tool, name: :find_tools)
  end

  it 'renders the same payload repeatedly without mutating history' do
    messages = [pending_search_message, search_result_message]
    original = messages.map(&:to_h)
    first = render(messages)

    expect(render(messages)).to eq(first)
    expect(messages.map(&:to_h)).to eq(original)
  end
end
