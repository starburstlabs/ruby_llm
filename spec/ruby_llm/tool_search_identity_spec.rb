# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Tool do
  include_context 'with configured RubyLLM'

  before do
    stub_const('CatalogFinder', Class.new(described_class) do
      description 'Finds tools for a task'
      searches_tools
      attr_reader :searches

      def initialize
        super
        @searches = []
      end

      def execute(topic:, limit: 3)
        @searches << { topic:, limit: }
        RubyLLM::ToolSearchResults.new(:lookup_order)
      end
    end)
    stub_const('LookupOrder', Class.new(described_class) do
      description 'Looks up an order'
      def execute(order_id:) = "Order #{order_id} shipped"
    end)
    stub_const('ToolSearch', Class.new(described_class) do
      description 'Searches an ordinary catalog'
      def execute(query:) = "Ordinary search: #{query}"
    end)
  end

  let(:finder) { CatalogFinder.new }
  let(:target) { LookupOrder.new }
  let(:chat) { RubyLLM.chat(model: model_for(:openai, :tool_namespace), provider: :openai) }

  describe '.searches_tools' do
    it 'gives a declared search tool the canonical name' do
      expect(CatalogFinder.tool_name).to eq('tool_search')
      expect(finder.name).to eq('tool_search')
      expect(finder.namespace).to be_nil
    end

    it 'inherits the role and canonical name without changing ordinary subclasses' do
      stub_const('OrderFinder', Class.new(CatalogFinder))
      stub_const('OrderLookup', Class.new(LookupOrder))

      expect(OrderFinder.searches_tools?).to be(true)
      expect(OrderFinder.new.name).to eq('tool_search')
      expect(OrderLookup.searches_tools?).to be(false)
      expect(OrderLookup.new.name).to eq('order_lookup')
    end

    it 'uses the canonical name after the derived name was read' do
      stub_const('CachedFinder', Class.new(described_class))
      instance = CachedFinder.new
      expect(CachedFinder.tool_name).to eq('cached_finder')
      expect(instance.name).to eq('cached_finder')

      CachedFinder.searches_tools

      expect(CachedFinder.tool_name).to eq('tool_search')
      expect(instance.name).to eq('tool_search')
    end

    it 'distinguishes the declared role from an ordinary tool named tool_search' do
      expect(CatalogFinder.searches_tools?).to be(true)
      expect(finder.searches_tools?).to be(true)
      expect(ToolSearch.tool_name).to eq('tool_search')
      expect(ToolSearch.searches_tools?).to be(false)
      expect(ToolSearch.new.searches_tools?).to be(false)
    end

    it 'preserves the declared execute arguments' do
      expect(finder.parameters_schema).to eq(
        'type' => 'object',
        'properties' => { 'topic' => { 'type' => 'string' }, 'limit' => { 'type' => 'string' } },
        'required' => ['topic'],
        'additionalProperties' => false,
        'strict' => true
      )
      expect(finder.call(topic: 'orders', limit: 2)).to be_a(RubyLLM::ToolSearchResults)
      expect(finder.searches).to eq([{ topic: 'orders', limit: 2 }])
    end
  end

  describe 'registration' do
    it 'rejects a namespace on a declared search tool' do
      CatalogFinder.namespace 'catalog'

      expect { chat.with_tools(finder) }.to raise_error(ArgumentError, /tool_search/)
    end

    it 'rejects an inherited namespace on a declared search tool' do
      stub_const('ApplicationTool', Class.new(described_class) { namespace 'catalog' })
      stub_const('InheritedFinder', Class.new(ApplicationTool) { searches_tools })

      expect { chat.with_tools(InheritedFinder) }.to raise_error(ArgumentError, /tool_search/)
    end

    it 'accepts an explicitly cleared inherited namespace' do
      stub_const('ApplicationTool', Class.new(described_class) { namespace 'catalog' })
      stub_const('InheritedFinder', Class.new(ApplicationTool) do
        namespace nil
        searches_tools
      end)
      search = InheritedFinder.new

      chat.with_tools(search, target, defer: true)

      expect(chat.tools).to eq(lookup_order: target, tool_search: search)
    end

    it 'rejects an incompatible instance name override' do
      finder.define_singleton_method(:name) { 'custom_finder' }

      expect { chat.with_tools(finder) }.to raise_error(ArgumentError, /tool_search/)
    end

    it 'rejects an incompatible class tool_name override' do
      CatalogFinder.define_singleton_method(:tool_name) { 'custom_finder' }

      expect { chat.with_tools(CatalogFinder) }.to raise_error(ArgumentError, /tool_search/)
    end

    it 'rejects a different name and namespace even when their qualified name is tool_search' do
      CatalogFinder.namespace 'tool'
      CatalogFinder.define_singleton_method(:tool_name) { 'search' }

      expect(finder.qualified_name).to eq('tool_search')
      expect { chat.with_tools(finder) }.to raise_error(ArgumentError, /tool_search/)
    end

    it 'rejects an ordinary tool replacing a declared search tool' do
      chat.with_tools(finder, target, defer: true)

      expect { chat.with_tools(ToolSearch) }.to raise_error(ArgumentError, /tool_search/)
      expect(chat.tools[:tool_search]).to be(finder)
    end

    it 'rejects a declared search tool replacing an ordinary tool' do
      ordinary = ToolSearch.new
      chat.with_tools(ordinary)

      expect { chat.with_tools(finder) }.to raise_error(ArgumentError, /tool_search/)
      expect(chat.tools[:tool_search]).to be(ordinary)
    end

    it 'replaces the finder with a different declared implementation' do
      stub_const('ReplacementFinder', Class.new(described_class) do
        searches_tools
        def execute = RubyLLM::ToolSearchResults.new(:lookup_order)
      end)
      replacement = ReplacementFinder.new
      chat.with_tools(finder, target, defer: true)

      chat.with_tools(replacement)

      expect(chat.tools).to eq(lookup_order: target, tool_search: replacement)
      expect(chat.deferred_tools).to eq(lookup_order: target)
    end

    it 'keeps the finder eager in a bulk registration despite class and instance deferral' do
      CatalogFinder.defer
      finder.defer

      chat.with_tools(finder, target, defer: true)

      expect(chat.deferred_tools).to eq(lookup_order: target)
      expect(chat.tools[:tool_search]).to be(finder)
    end

    it 'keeps the finder eager through an Agent bulk declaration' do
      agent = Class.new(RubyLLM::Agent)
      agent.model model_for(:openai, :tool_namespace), provider: :openai
      agent.tools CatalogFinder, LookupOrder, defer: true

      expect(agent.chat.deferred_tools.keys).to eq([:lookup_order])
      expect(agent.chat.tools.keys).to eq(%i[lookup_order tool_search])
    end

    it 'executes an ordinary tool named tool_search without the search role' do
      chat.with_tools(ToolSearch)
      call = RubyLLM::ToolCall.new(id: 'ordinary_1', name: 'tool_search', arguments: { 'query' => 'orders' })
      chat.add_message(role: :assistant, content: nil, tool_calls: { call.id => call })

      chat.run_tools

      expect(chat.messages.last.content).to eq('Ordinary search: orders')
      expect(chat.render[:tools].map { |tool| tool.slice(:type, :name) })
        .to eq([{ type: 'function', name: 'tool_search' }])
      expect(chat.render[:input].last).to include(type: 'function_call_output', call_id: 'ordinary_1')
    end
  end

  it 'executes a parsed bare client search through the canonical finder' do
    item = { 'type' => 'tool_search_call', 'execution' => 'client', 'id' => 'item_1',
             'call_id' => 'search_1', 'arguments' => { 'topic' => 'orders', 'limit' => '2' }, 'status' => 'completed' }
    body = { 'output' => [item], 'model' => model_for(:openai, :tool_namespace),
             'status' => 'completed', 'usage' => {} }
    stub_request(:post, 'https://api.openai.com/v1/responses').to_return(
      status: 200, headers: { 'Content-Type' => 'application/json' }, body: JSON.generate(body)
    )
    chat.with_tools(finder, target, defer: true).ask_later('Find order 42')

    response = chat.generate
    call = response.tool_calls.values.sole
    chat.run_tools

    expect(call).to have_attributes(name: 'tool_search', namespace: nil, provider_call_id: 'search_1',
                                    arguments: { 'topic' => 'orders', 'limit' => '2' })
    expect(finder.searches).to eq([{ topic: 'orders', limit: '2' }])
    expect(chat.messages.last).to have_attributes(content: '{"loaded":["lookup_order"]}', tool_call_id: call.id)
    expect(chat.render[:input].last).to include(type: 'tool_search_output', call_id: 'search_1',
                                                tools: [hash_including(name: 'lookup_order')])
  end
end
