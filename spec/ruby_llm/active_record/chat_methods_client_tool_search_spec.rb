# frozen_string_literal: true

require 'rails_helper'
require_relative '../../support/client_tool_search_helpers'

RSpec.describe RubyLLM::ActiveRecord::ChatMethods do
  include_context 'with client tool search'

  let(:call_id) { "search_#{SecureRandom.hex(6)}" }
  let(:chat) { Chat.create!(model: 'gpt-5.4').with_tools(finder, target, defer: true) }

  it 'persists the original Ruby name, call ID, arguments, raw items, and neutral results across reloads' do
    chat.add_message(pending_search_message(raw: [client_search_item(call_id:)], call_id:))
    chat.run_tools
    reloaded = Chat.find(chat.id).with_tools(finder, target, defer: true)
    messages = reloaded.to_llm.messages
    call = messages.first.tool_calls.fetch(call_id)

    expect(call).to have_attributes(name: 'find_tools', arguments: { 'query' => 'orders' })
    expect(messages.first.raw_content).to eq([client_search_item(call_id:)])
    expect(messages.last.content).to eq('{"tool_search_results":["lookup_order"]}')
    expect(messages.last.tool_call_id).to eq(call_id)
    expect(reloaded.render[:input].last).to include(type: 'tool_search_output', call_id: call_id)
    expect(RubyLLM::ActiveRecord::ToolCall.find_by(tool_call_id: call_id).result.content)
      .to eq(messages.last.content)
  end

  it 'leaves invalid fresh results pending in the database and retries after reloading' do
    finder.result = RubyLLM::ToolSearchResults.new(:retired)
    chat.add_message(pending_search_message(raw: [client_search_item(call_id:)], call_id:))

    expect { chat.run_tools }.to raise_error(ArgumentError, /retired/)
    expect(chat.messages.count).to eq(1)
    expect(RubyLLM::ActiveRecord::ToolCall.find_by(tool_call_id: call_id).result).to be_nil

    finder.result = RubyLLM::ToolSearchResults.new(target)
    reloaded = Chat.find(chat.id).with_tools(finder, target, defer: true)
    reloaded.run_tools

    expect(finder.queries).to eq(%w[orders orders])
    expect(reloaded.messages.count).to eq(2)
    expect(reloaded.render[:input].last[:tools].map { |tool| tool[:name] }).to eq(['lookup_order'])
  end

  it 'omits a historical retired tool after a reload and retains its stored result' do
    chat.add_message(pending_search_message(raw: [client_search_item(call_id:)], call_id:))
    chat.run_tools
    reloaded = Chat.find(chat.id).with_tools(finder)
    allow(RubyLLM.logger).to receive(:warn)

    expect(reloaded.render[:input].last[:tools]).to eq([])
    expect(RubyLLM.logger).to have_received(:warn).with(/lookup_order/)
    expect(reloaded.messages.last.content).to eq('{"tool_search_results":["lookup_order"]}')
  end
end
