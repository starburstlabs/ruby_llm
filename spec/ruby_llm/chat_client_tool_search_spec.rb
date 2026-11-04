# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/client_tool_search_helpers'

RSpec.describe RubyLLM::Chat do
  include_context 'with client tool search'

  let(:chat) { described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true) }

  it 'keeps search roles eager when all tools are deferred together' do
    expect(chat.deferred_tools.keys).to eq([:lookup_order])
    expect(chat.render[:tools]).to include(hash_including(type: 'tool_search', execution: 'client'))
    expect(chat.tools[:find_tools]).to be(finder)
  end

  it 'keeps a search role eager when its class declares defer' do
    FindTools.defer
    configured = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(FindTools)

    expect(configured.deferred_tools).to eq({})
    expect(configured.render[:tools].first).not_to have_key(:defer_loading)
  end

  it 'keeps the search role eager through an Agent bulk declaration' do
    agent = Class.new(RubyLLM::Agent)
    agent.model 'gpt-5.4', provider: :openai
    agent.tools FindTools, LookupOrder, defer: true

    expect(agent.chat.deferred_tools.keys).to eq([:lookup_order])
  end

  it 'executes the registered search tool and stores only the neutral envelope' do
    chat.add_message(pending_search_message)
    chat.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(chat.messages.last.content).to eq('{"tool_search_results":["lookup_order"]}')
    expect(chat.messages.last.raw_content).to be_nil
  end

  %i[missing find_tools].each do |name|
    it "rejects fresh #{name} results before adding a message, allowing retry" do
      finder.result = RubyLLM::ToolSearchResults.new(name)
      chat.add_message(pending_search_message)

      expect { chat.run_tools }.to raise_error(ArgumentError, /unavailable tools/)
      expect(chat.messages.size).to eq(1)
      finder.result = RubyLLM::ToolSearchResults.new(target)
      chat.run_tools
      expect(finder.queries).to eq(%w[orders orders])
      expect(chat.messages.last.tool_call_id).to eq('search_1')
    end
  end

  it 'rejects fresh typed results from an unmarked tool' do
    allow(target).to receive(:call).and_return(RubyLLM::ToolSearchResults.new)
    call = RubyLLM::ToolCall.new(id: 'order_1', name: target.name, arguments: { 'order_id' => '42' })
    chat.add_message(role: :assistant, content: nil, tool_calls: { call.id => call })

    expect { chat.run_tools }.to raise_error(ArgumentError, /Only tools declaring searches_tools/)
    expect(chat.messages.size).to eq(1)
  end

  it 'accepts MCP aliases and rejects tools hidden from the model' do
    server = instance_double(RubyLLM::MCP)
    definition = { 'name' => 'lookup', 'description' => 'Find an order', 'inputSchema' => { 'type' => 'object' } }
    visible = RubyLLM::MCP::Tool.new(server, definition, as: 'orders_lookup')
    hidden = RubyLLM::MCP::Tool.new(server, definition.merge('_meta' => {
                                                               'ui' => { 'resourceUri' => 'ui://orders',
                                                                         'visibility' => ['app'] }
                                                             }), as: 'orders_private')
    chat.with_tools(visible, hidden)
    finder.result = RubyLLM::ToolSearchResults.new(visible)
    chat.add_message(pending_search_message)
    chat.run_tools
    expect(JSON.parse(chat.messages.last.content)).to eq('tool_search_results' => ['orders_lookup'])

    finder.result = RubyLLM::ToolSearchResults.new(hidden)
    chat.add_message(pending_search_message(call_id: 'search_2'))
    expect { chat.run_tools }.to raise_error(ArgumentError, /orders_private/)
  end

  it 'rejects ambiguous search handlers' do
    other = FindTools.new
    allow(other).to receive(:name).and_return('another_search')
    chat.with_tools(other)

    expect { chat.render }.to raise_error(ArgumentError, /only one tool/)
  end

  it 'keeps ordinary eager fallback on Chat Completions' do
    chat.with_model(model_for(:openai), provider: :openai, protocol: :chat_completions)
    chat.add_message(pending_search_message)
    chat.run_tools
    payload = chat.render

    expect(payload[:tools].map { |tool| tool.dig(:function, :name) }).to eq(%w[find_tools lookup_order])
    expect(payload.to_json).not_to include('defer_loading')
    expect(payload[:messages].last[:content]).to eq('{"tool_search_results":["lookup_order"]}')
  end
end
