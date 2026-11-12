# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/application_tool_search_helpers'

RSpec.describe RubyLLM::Chat do
  include_context 'with application tool search'

  let(:chat) { described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target, defer: true) }

  it 'keeps the search tool eager when all tools are deferred together' do
    expect(chat.deferred_tools.keys).to eq([:lookup_order])
    expect(chat.tools[:find_tools]).to be(finder)
  end

  it 'keeps the search tool eager when its class declares defer' do
    FindTools.defer
    configured = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(FindTools, target, defer: true)

    expect(configured.deferred_tools.keys).to eq([:lookup_order])
  end

  it 'keeps the search tool eager through an Agent bulk declaration' do
    agent = Class.new(RubyLLM::Agent)
    agent.model 'gpt-5.4', provider: :openai
    agent.tools FindTools, LookupOrder, defer: true

    expect(agent.chat.deferred_tools.keys).to eq([:lookup_order])
    expect(agent.chat.tools.keys).to eq(%i[lookup_order find_tools])
  end

  it 'offers the search tool only while a tool is deferred, without recursing' do
    plain = described_class.new(model: 'gpt-5.4', provider: :openai).with_tools(finder, target)

    expect(plain.tools.keys).to eq([:lookup_order])
    expect(plain.with_tools(target, defer: true).tools.keys).to eq(%i[lookup_order find_tools])
  end

  it 'executes the search tool and stores what it loaded' do
    chat.add_message(pending_search_message)
    chat.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(chat.messages.last.content).to eq('{"loaded":["lookup_order"]}')
    expect(chat.messages.last.raw_content).to be_nil
  end

  it 'rejects a second search tool' do
    other = Class.new(FindTools) { def self.tool_name = 'another_search' }

    expect { chat.with_tools(other) }.to raise_error(ArgumentError, 'find_tools and another_search both search ' \
                                                                    'tools. Register one')
  end

  it 'names the search tool to a model that calls a tool it has not loaded' do
    call = RubyLLM::ToolCall.new(id: 'call_1', name: 'lookup_orders', arguments: {})
    chat.add_message(role: :assistant, content: nil, tool_calls: { call.id => call })
    chat.run_tools

    expect(chat.messages.last.content).to include('Did you mean `lookup_order`?', 'Find tools with `find_tools`.')
  end

  describe 'on a provider without tool search' do
    let(:chat) do
      described_class.new(model: model_for(:gemini), provider: :gemini).with_tools(finder, target, defer: true)
    end

    def declared(chat)
      chat.render[:tools].flat_map { |tool| tool[:functionDeclarations] }.map { |declaration| declaration[:name] }
    end

    it 'replaces tool_search with the search tool' do
      chat.add_message(role: :user, content: 'Hi')

      expect(chat.tools.keys).to eq(%i[lookup_order find_tools])
      expect(declared(chat)).to eq(%w[find_tools])
    end

    it 'appends the tools the search loaded, after the others' do
      chat.add_message(role: :user, content: 'Find order 42')
      chat.add_message(pending_search_message)
      chat.run_tools

      expect(declared(chat)).to eq(%w[find_tools lookup_order])
    end

    it 'keeps a search result whole when the file store saves long results as files' do
      chat.with_file_store(:memory, inline_limit: 10)
      chat.add_message(role: :user, content: 'Find order 42')
      chat.add_message(pending_search_message)
      chat.run_tools

      expect(chat.messages.last.content).to eq('{"loaded":["lookup_order"]}')
      expect(declared(chat)).to eq(%w[list_files read_file search_files write_file edit_file find_tools lookup_order])
    end

    it 'still loads what the built-in tool_search found before the search tool was registered' do
      call = RubyLLM::ToolCall.new(id: 'search_0', name: 'tool_search', arguments: { 'query' => 'orders' })
      chat.add_message(role: :user, content: 'Hi')
      chat.add_message(role: :assistant, content: nil, tool_calls: { call.id => call })
      chat.add_message(role: :tool, content: { loaded: %w[lookup_order], tools: [] }.to_json, tool_call_id: 'search_0')

      expect(declared(chat)).to eq(%w[find_tools lookup_order])
    end

    it 'forgets what a renamed search tool loaded, except tools the model went on to call' do
      chat.add_message(role: :user, content: 'Hi')
      chat.add_message(pending_search_message(name: 'old_search'))
      chat.add_message(search_result_message)

      expect(declared(chat)).to eq(%w[find_tools])

      call = RubyLLM::ToolCall.new(id: 'order_1', name: 'lookup_order', arguments: { 'order_id' => '42' })
      chat.add_message(role: :assistant, content: nil, tool_calls: { call.id => call })
      chat.add_message(role: :tool, content: 'Order 42 shipped', tool_call_id: 'order_1')

      expect(declared(chat)).to eq(%w[find_tools lookup_order])
    end

    it 'counts tokens with the tools of the request' do
      counted = nil
      stub_request(:post, /:countTokens/).to_return do |request|
        counted = JSON.parse(request.body)
        { status: 200, headers: { 'Content-Type' => 'application/json' }, body: { totalTokens: 7 }.to_json }
      end
      chat.add_message(role: :user, content: 'Find order 42')
      chat.add_message(pending_search_message)
      chat.run_tools
      chat.count_tokens

      declarations = counted.dig('generateContentRequest', 'tools').flat_map { |tool| tool['functionDeclarations'] }
      expect(declarations.map { |declaration| declaration['name'] }).to eq(%w[find_tools lookup_order])
    end
  end

  it 'replaces tool_search on Chat Completions and appends what it loaded' do
    chat.with_model(model_for(:openai), provider: :openai, protocol: :chat_completions)
    chat.add_message(role: :user, content: 'Find order 42')
    chat.add_message(pending_search_message)
    chat.run_tools
    payload = chat.render

    expect(payload[:tools].map { |tool| tool.dig(:function, :name) }).to eq(%w[find_tools lookup_order])
    expect(payload.to_json).not_to include('defer_loading', 'tool_search')
    expect(payload[:messages].last[:content]).to eq('{"loaded":["lookup_order"]}')
  end

  it 'sends the search tool as a plain function on xAI' do
    grok = described_class.new(model: model_for(:xai), provider: :xai).with_tools(finder, target, defer: true)
    grok.add_message(role: :user, content: 'Hi')
    tools = grok.render[:tools]

    expect(tools.map { |tool| tool.slice(:type, :name) }).to eq([{ type: 'function', name: 'find_tools' }])
    expect(tools.to_json).not_to include('defer_loading', 'execution')
  end
end
