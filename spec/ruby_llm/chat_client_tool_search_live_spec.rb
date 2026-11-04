# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/client_tool_search_helpers'

RSpec.describe RubyLLM::Chat, :live do
  include_context 'with client tool search'

  describe 'client tool search' do
    each_model(TOOL_SEARCH_MODELS) do |provider, model|
      context "with #{provider}/#{model}" do
        before { skip_without_cassette_or_key("#{provider.to_s.upcase}_API_KEY") }

        let(:chat) { described_class.new(model:, provider:).with_tools(finder, target, defer: true) }
        let(:question) { 'First search for an order lookup tool, then use it to check order 42. What is its status?' }

        it 'searches with the application handler and executes the matching tool' do
          response = chat.ask(question)

          expect(finder.queries).not_to be_empty
          expect(response.content.downcase).to include('42', 'shipped')
          result = chat.messages.find { |message| RubyLLM::ToolSearchResults.from_content(message.content) }
          expect(RubyLLM::ToolSearchResults.from_content(result.content).names).to eq(['lookup_order'])
          expect(chat.messages.select(&:tool_call?).flat_map { |message| message.tool_calls.values.map(&:name) })
            .to include('find_tools', 'lookup_order')
        end

        it 'searches and executes the matching tool while streaming' do
          chunks = []
          response = chat.ask(question) { |chunk| chunks << chunk }

          expect(finder.queries).not_to be_empty
          expect(response.content.downcase).to include('42', 'shipped')
          expect(chunks).not_to be_empty
          expect(chat.messages.count(&:tool_result?)).to be >= 2
        end
      end
    end
  end
end
