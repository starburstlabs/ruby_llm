# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::ToolSearchResults do
  before do
    stub_const('LookupOrder', Class.new(RubyLLM::Tool))
    stub_const('CustomerLookup', Class.new(RubyLLM::Tool) do
      namespace 'crm'
      def self.tool_name = 'customer_lookup'
    end)
  end

  it 'normalizes names, tool classes, and instances in order without duplicates' do
    results = described_class.new(:weather, LookupOrder, 'weather', LookupOrder.new, :calendar)

    expect(results.names).to eq(%w[weather lookup_order calendar])
  end

  it 'names a namespaced tool as the chat registers it, from its class or an instance' do
    expect(described_class.new(CustomerLookup, CustomerLookup.new).names).to eq(%w[crm_customer_lookup])
  end

  it 'names the base tool class and instance as the chat registers them' do
    expect(described_class.new(RubyLLM::Tool, RubyLLM::Tool.new).names).to eq([RubyLLM::Tool.new.qualified_name])
  end

  it 'accepts empty results and serializes them through the ordinary tool result path' do
    content, attachments = RubyLLM::Tool.split_result(described_class.new)

    expect(content).to eq('{"loaded":[]}')
    expect(described_class.from_content(content).names).to eq([])
    expect(attachments).to eq([])
  end

  it 'round trips its content without colliding with document search results' do
    content, = RubyLLM::Tool.split_result(described_class.new(LookupOrder))

    expect(content).to eq('{"loaded":["lookup_order"]}')
    expect(described_class.from_content(content).names).to eq(['lookup_order'])
    expect(RubyLLM::SearchResults.from_content(content)).to be_nil
  end

  it 'reads what the built-in tool_search stores' do
    found = { loaded: %w[weather_lookup], tools: [{ name: 'weather_lookup', description: 'Weather' }] }.to_json
    nothing = { loaded: [], error: 'No tools match calendar' }.to_json

    expect(described_class.from_content(found).names).to eq(%w[weather_lookup])
    expect(described_class.from_content(nothing).names).to eq([])
  end

  it 'reads every member of a loaded list as a name' do
    expect(described_class.from_content('{"loaded":[1,"a"]}').names).to eq(%w[1 a])
  end

  it 'provides a concise inspect' do
    expect(described_class.new(:weather).inspect).to eq('#<RubyLLM::ToolSearchResults names: ["weather"]>')
  end

  [nil, '{}', 'null', '[]', 'text', '{invalid', '{"loaded":"a"}', ' {"loaded":["a"]}'].each do |content|
    it "ignores content that is not a result, #{content.inspect}" do
      expect(described_class.from_content(content)).to be_nil
    end
  end

  [Object, Object.new, 42, nil].each do |tool|
    it "rejects invalid tool entries, #{tool.inspect}" do
      expect { described_class.new(tool) }.to raise_error(ArgumentError, /takes Tool classes/)
    end
  end

  it 'inherits the search role through tool classes and wrappers' do
    base = Class.new(RubyLLM::Tool) { searches_tools }
    child = Class.new(base)
    deferred = RubyLLM::Tool::Deferred.new(child.new)

    expect(RubyLLM::Tool.new.searches_tools?).to be(false)
    expect(child.new.searches_tools?).to be(true)
    expect(deferred.searches_tools?).to be(true)
    expect(RubyLLM::Tool::Flattened.wrap(deferred).searches_tools?).to be(true)
  end

  describe '#within' do
    it 'keeps the deferred tools of a request it names, in the order named' do
      tools = { lookup_order: RubyLLM::Tool::Deferred.new(LookupOrder.new),
                weather: RubyLLM::Tool::Deferred.new(RubyLLM::Tool.new), current_time: RubyLLM::Tool.new }

      expect(described_class.new(:weather, :current_time, :lookup_order, :missing).within(tools))
        .to eq(weather: tools[:weather], lookup_order: tools[:lookup_order])
    end
  end
end
