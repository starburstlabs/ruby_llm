# frozen_string_literal: true

require 'rails_helper'
require_relative '../../support/application_tool_search_helpers'

RSpec.describe RubyLLM::ActiveRecord::ChatMethods do
  include_context 'with application tool search'

  let(:call_id) { "search_#{SecureRandom.hex(6)}" }
  let(:chat) { Chat.create!(model: 'gpt-5.4').with_tools(finder, target, defer: true) }

  it 'persists the search by its Ruby name, call ID, arguments, and result, and replays it natively after a reload' do
    chat.add_message(pending_search_message(call_id:))
    chat.run_tools
    reloaded = Chat.find(chat.id).with_tools(finder, target, defer: true)
    messages = reloaded.to_llm.messages
    input = reloaded.render[:input]

    expect(messages.first.tool_calls.fetch(call_id)).to have_attributes(name: 'find_tools',
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
end
