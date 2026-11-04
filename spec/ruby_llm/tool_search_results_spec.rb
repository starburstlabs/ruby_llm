# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::ToolSearchResults do
  before do
    stub_const('LookupOrder', Class.new(RubyLLM::Tool))
  end

  it 'normalizes names, tool classes, and instances in order without duplicates' do
    results = described_class.new(:weather, LookupOrder, 'weather', LookupOrder.new, :calendar)

    expect(results.names).to eq(%w[weather lookup_order calendar])
  end

  it 'accepts empty results and serializes them through the ordinary tool result path' do
    content, attachments = RubyLLM::Tool.split_result(described_class.new)

    expect(JSON.parse(content)).to eq('tool_search_results' => [])
    expect(described_class.from_content(content).names).to eq([])
    expect(attachments).to eq([])
  end

  it 'round trips the neutral envelope without colliding with document search results' do
    content, = RubyLLM::Tool.split_result(described_class.new(LookupOrder))

    expect(described_class.from_content(content).names).to eq(['lookup_order'])
    expect(RubyLLM::SearchResults.from_content(content)).to be_nil
  end

  it 'provides a concise inspect' do
    expect(described_class.new(:weather).inspect).to eq('#<RubyLLM::ToolSearchResults names: ["weather"]>')
  end

  [nil, '{}', 'null', '[]', 'text', '{invalid', '{"tool_search_results":[1]}'].each do |content|
    it "ignores content that is not a result envelope, #{content.inspect}" do
      expect(described_class.from_content(content)).to be_nil
    end
  end

  [Object, Object.new, 42, nil].each do |tool|
    it "rejects invalid tool entries, #{tool.inspect}" do
      expect { described_class.new(tool) }.to raise_error(ArgumentError, /require tool names/)
    end
  end

  it 'inherits the search role through tool classes and deferred wrappers' do
    base = Class.new(RubyLLM::Tool) { searches_tools }
    child = Class.new(base)

    expect(RubyLLM::Tool.new.searches_tools?).to be(false)
    expect(child.new.searches_tools?).to be(true)
    expect(RubyLLM::Tool::Deferred.new(child.new).searches_tools?).to be(true)
  end
end
