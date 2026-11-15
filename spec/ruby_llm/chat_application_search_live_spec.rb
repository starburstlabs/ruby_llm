# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/application_tool_search_helpers'

RSpec.describe RubyLLM::Chat, :live do
  include_context 'with application tool search'

  let(:question) { 'First search for an order lookup tool, then use it to check order 42. What is its status?' }

  def provider_key(provider)
    provider == :bedrock ? 'AWS_ACCESS_KEY_ID' : "#{provider.to_s.upcase}_API_KEY"
  end

  def calls(chat)
    chat.messages.select(&:tool_call?).flat_map { |message| message.tool_calls.values }
  end

  shared_examples 'an application tool search' do
    it 'searches with the application tool and calls what it loaded' do
      response = chat.ask(question)

      expect(finder.queries).not_to be_empty
      expect(response.content.downcase).to include('shipped')
      expect(chat.messages.map(&:content)).to include('{"loaded":["lookup_order"]}')
      expect(calls(chat)).to include(have_attributes(name: 'tool_search'),
                                     have_attributes(name: 'lookup_order', arguments: { 'order_id' => '42' }))
    end

    it 'searches with the application tool and calls what it loaded while streaming' do
      chunks = []

      response = chat.ask(question) { |chunk| chunks << chunk }

      expect(finder.queries).not_to be_empty
      expect(response.content.downcase).to include('shipped')
      expect(calls(chat)).to include(have_attributes(name: 'lookup_order', arguments: { 'order_id' => '42' }))
      expect(chunks).not_to be_empty
    end
  end

  describe 'on a provider with tool search' do
    each_model(TOOL_SEARCH_MODELS) do |provider, model|
      context "with #{provider}/#{model}" do
        before { skip_without_cassette_or_key(provider_key(provider)) }

        let(:chat) { described_class.new(model:, provider:).with_tools(finder, target, defer: true) }

        it_behaves_like 'an application tool search'
      end
    end
  end

  describe 'on a provider without tool search' do
    model = model_for(:gemini, :application_tool_search)

    context "with gemini/#{model}" do
      before { skip_without_cassette_or_key('GEMINI_API_KEY') }

      let(:chat) { described_class.new(model:, provider: :gemini).with_tools(finder, target, defer: true) }

      it_behaves_like 'an application tool search'
    end
  end
end
