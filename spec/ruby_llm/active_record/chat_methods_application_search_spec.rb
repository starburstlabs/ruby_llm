# frozen_string_literal: true

require 'rails_helper'
require_relative '../../support/application_tool_search_helpers'

RSpec.describe RubyLLM::ActiveRecord::ChatMethods do
  include_context 'with application tool search'

  let(:call_id) { "search_#{SecureRandom.hex(6)}" }
  let(:chat) { Chat.create!(model: 'gpt-5.4').with_tools(finder, target, defer: true) }

  it 'persists the canonical search name, call ID, arguments, and result, and replays it after a reload' do
    chat.add_message(pending_search_message(call_id:))
    chat.run_tools
    reloaded = Chat.find(chat.id).with_tools(finder, target, defer: true)
    messages = reloaded.to_llm.messages
    input = reloaded.render[:input]

    expect(messages.first.tool_calls.fetch(call_id)).to have_attributes(name: 'tool_search',
                                                                        arguments: { 'query' => 'orders' })
    expect(messages.first.raw_content).to be_nil
    expect(messages.last).to have_attributes(content: '{"loaded":["lookup_order"]}', tool_call_id: call_id)
    expect(input.map { |item| item[:type] }).to eq(%w[tool_search_call tool_search_output])
    expect(input.last).to include(call_id:, tools: [hash_including(name: 'lookup_order', defer_loading: true)])
    expect(RubyLLM::ActiveRecord::ToolCall.find_by(tool_call_id: call_id).result.content)
      .to eq(messages.last.content)
  end

  it 'drops a loaded tool that is no longer registered after a reload, keeping the stored result' do
    chat.add_message(pending_search_message(call_id:))
    chat.run_tools
    lookup_customer = Class.new(RubyLLM::Tool) do
      description 'Looks up a customer'
      def self.tool_name = 'lookup_customer'
      def execute = 'Ada'
    end
    reloaded = Chat.find(chat.id).with_tools(finder, lookup_customer, defer: true)

    expect(reloaded.render[:input].last).to include(type: 'tool_search_output', call_id:, tools: [])
    expect(reloaded.messages.last.content).to eq('{"loaded":["lookup_order"]}')
  end

  it 'executes a stored canonical search after its tools are reapplied' do
    unconfigured = Chat.create!(model: model_for(:openai, :tool_namespace))
    raw = [client_search_item(call_id:)]
    protocol = RubyLLM::Protocols::Responses.allocate
    unconfigured.add_completion(protocol.send(:parse_completion_body, responses_body(raw), raw: nil))

    expect(RubyLLM::ActiveRecord::ToolCall.find_by!(provider_call_id: call_id).name).to eq('tool_search')
    reloaded = Chat.find(unconfigured.id).with_tools(finder, target, defer: true)
    reloaded.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(reloaded.messages.last.content).to eq('{"loaded":["lookup_order"]}')
    again = Chat.find(unconfigured.id).with_tools(finder, target, defer: true)
    expect(again.render[:input].last).to include(type: 'tool_search_output', call_id:,
                                                 tools: [hash_including(name: 'lookup_order')])
  end

  it 'persists an out-of-band search and executes it after approval and a reload' do
    FindTools.requires_approval
    raw = [client_search_item(call_id:)]
    protocol = RubyLLM::Protocols::Responses.allocate
    message = protocol.send(:parse_completion_body, responses_body(raw), raw: nil)
    seen = []
    chat.after_message { |reply| seen << reply.tool_calls.values.sole.name }

    delivered = chat.add_completion(message)

    stored = RubyLLM::ActiveRecord::ToolCall.find_by!(tool_call_id: delivered.tool_calls.values.sole.id)
    expect(stored.name).to eq('tool_search')
    expect(stored.provider_call_id).to eq(call_id)
    expect(stored.approval).to eq('pending')
    expect(seen).to eq(['tool_search'])
    reloaded = Chat.find(chat.id).with_tools(finder, target, defer: true)
    expect(reloaded.to_llm.messages.last.raw_content).to eq(raw)
    reloaded.run_tools
    expect(finder.queries).to be_empty
    question = reloaded.pending_questions.find(&:approval?)
    expect(question.tool_call).to have_attributes(name: 'tool_search', provider_call_id: call_id)
    reloaded.approve(question)
    reloaded.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(reloaded.messages.last.content).to eq('{"loaded":["lookup_order"]}')
    expect(reloaded.render[:input].last).to include(type: 'tool_search_output', call_id:,
                                                    tools: [hash_including(name: 'lookup_order')])
  end
end
