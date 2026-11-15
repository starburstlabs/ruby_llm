# frozen_string_literal: true

require 'spec_helper'
require_relative '../support/application_tool_search_helpers'

RSpec.describe RubyLLM::Batch do
  include_context 'with application tool search'

  let(:chat) do
    RubyLLM.chat(model: model_for(:openai, :tool_namespace), provider: :openai).with_tools(finder, target, defer: true)
           .ask_later('Find order 42')
  end
  let(:ordinary) do
    Class.new(RubyLLM::Tool) do
      attr_reader :queries

      def self.tool_name = 'tool_search'

      def initialize
        super
        @queries = []
      end

      def execute(query:)
        @queries << query
        "Ordinary search: #{query}"
      end
    end.new
  end

  def collect(output, completed: true)
    body = responses_body(output)
    body['usage'] = { 'input_tokens' => 5, 'output_tokens' => 2 }
    jsonl = JSON.generate(custom_id: '0', response: { status_code: 200, body: })
    stub_request(:get, 'https://api.openai.com/v1/batches/batch_search')
      .to_return(json_response('id' => 'batch_search', 'status' => 'completed',
                               'endpoint' => '/v1/responses', 'output_file_id' => 'file_search'))
    allow(chat.provider).to receive(:download_file).with('file_search').and_return("#{jsonl}\n")
    described_class.new(provider: chat.provider, chats: [chat], id: 'batch_search', batch_protocol: :responses,
                        raw_status: completed ? 'completed' : 'in_progress', completed:)
  end

  def ordinary_call(name: 'tool_search')
    { 'type' => 'function_call', 'id' => 'item_ordinary', 'call_id' => 'ordinary_1', 'name' => name,
      'arguments' => '{"query":"ordinary"}', 'status' => 'completed' }
  end

  it 'delivers the canonical finder to callbacks and executes it with native replay and stable IDs' do
    seen = []
    chat.after_message do |message|
      seen << message.tool_calls.map { |call| [call.name, call.namespace, call.provider_call_id] }
    end
    batch = collect([client_search_item], completed: false)

    message = batch.messages.first
    call = message.tool_calls.values.sole

    expect(call).to have_attributes(name: 'tool_search', namespace: nil, provider_call_id: 'search_1',
                                    arguments: { 'query' => 'orders' })
    expect(call.id).not_to eq('search_1')
    expect(message).to equal(chat.messages.last)
    expect(chat.usage_entries.first.message).to equal(message)
    expect(seen).to eq([[['tool_search', nil, 'search_1']]])
    expect(message.raw_content).to eq([client_search_item])
    expect(message.tokens).to have_attributes(input: 5, output: 2)
    expect(batch.messages.first.tool_calls.keys).to eq([call.id])
    expect(batch.messages.first).to equal(message)
    expect(chat.messages.count(&:tool_call?)).to eq(1)
    expect(seen.size).to eq(1)
    expect(chat.usage_entries.size).to eq(1)

    chat.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(chat.messages.last).to have_attributes(content: '{"loaded":["lookup_order"]}',
                                                  tool_call_id: call.id, provider_call_id: 'search_1')
    expect(chat.render[:input][1]).to eq(client_search_item)
    expect(chat.render[:input].last).to include(type: 'tool_search_output', call_id: 'search_1',
                                                tools: [hash_including(name: 'lookup_order')])
  end

  it 'keeps client search and an ordinary function distinct in the same batch answer' do
    ordinary.define_singleton_method(:name) { 'search_catalog' }
    chat.with_tools(ordinary)
    function_call = ordinary_call(name: 'search_catalog')
    message = collect([client_search_item, function_call]).messages.first

    expect(message.tool_calls.values).to contain_exactly(
      have_attributes(name: 'tool_search', namespace: nil, provider_call_id: 'search_1'),
      have_attributes(name: 'search_catalog', namespace: nil, provider_call_id: 'ordinary_1')
    )

    chat.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(ordinary.queries).to eq(['ordinary'])
    expect(message.raw_content).to eq([client_search_item, function_call])
    input = chat.render[:input]
    expect(input[1, 2]).to eq([client_search_item, function_call])
    expect(input).to include(hash_including(type: 'tool_search_output', call_id: 'search_1'),
                             hash_including(type: 'function_call_output', call_id: 'ordinary_1',
                                            output: 'Ordinary search: ordinary'))
  end

  it 'executes an ordinary tool_search in a chat without a finder' do
    chat.with_tools(nil).with_tools(ordinary)
    message = collect([ordinary_call]).messages.first
    chat.run_tools

    expect(message.tool_calls.values.sole).to have_attributes(name: 'tool_search', namespace: nil,
                                                              provider_call_id: 'ordinary_1')
    expect(finder.queries).to be_empty
    expect(ordinary.queries).to eq(['ordinary'])
    expect(chat.messages.last.content).to eq('Ordinary search: ordinary')
    expect(chat.render[:input].last).to include(type: 'function_call_output', call_id: 'ordinary_1')
  end

  it 'executes a delivered canonical search when tools are reapplied' do
    chat.with_tools(nil)
    collect([client_search_item]).messages
    expect(chat.messages.last.tool_calls.values.sole.name).to eq('tool_search')

    chat.with_tools(finder, target, defer: true)
    chat.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(chat.messages.find(&:tool_call?).tool_calls.values.sole)
      .to have_attributes(name: 'tool_search', namespace: nil, provider_call_id: 'search_1')
    expect(chat.messages.last.content).to eq('{"loaded":["lookup_order"]}')
  end

  it 'executes a namespaced target loaded by the canonical search' do
    LookupOrder.namespace 'catalog'
    finder.result = RubyLLM::ToolSearchResults.new(:catalog_lookup_order)
    collect([client_search_item]).messages
    chat.run_tools
    call = { 'type' => 'function_call', 'id' => 'item_order', 'call_id' => 'order_1', 'name' => 'lookup_order',
             'namespace' => 'catalog', 'arguments' => '{"order_id":"42"}', 'status' => 'completed' }

    message = collect([call]).messages.first
    chat.run_tools

    expect(message.tool_calls.values.sole).to have_attributes(name: 'lookup_order', namespace: 'catalog',
                                                              provider_call_id: 'order_1')
    expect(finder.queries).to eq(['orders'])
    expect(chat.messages.last.content).to eq('Order 42 shipped')
  end

  it 'executes a deferred function returned under its search namespace' do
    collect([client_search_item]).messages
    chat.run_tools
    call = { 'type' => 'function_call', 'id' => 'item_order', 'call_id' => 'order_1', 'name' => 'lookup_order',
             'namespace' => 'lookup_order', 'arguments' => '{"order_id":"42"}', 'status' => 'completed' }

    message = collect([call]).messages.first

    expect(message).to equal(chat.messages.last)
    expect(chat.usage_entries.last.message).to equal(message)

    chat.run_tools

    expect(message.tool_calls.values.sole).to have_attributes(name: 'lookup_order', namespace: nil,
                                                              provider_call_id: 'order_1')
    expect(chat.messages.last.content).to eq('Order 42 shipped')
  end

  it 'normalizes a delivered searched function after tools are reapplied' do
    chat.with_tools(nil)
    collect([client_search_item]).messages
    chat.with_tools(finder, target, defer: true).run_tools
    chat.with_tools(nil)
    function_call = { 'type' => 'function_call', 'id' => 'item_order', 'call_id' => 'order_1',
                      'name' => 'lookup_order', 'namespace' => 'lookup_order',
                      'arguments' => '{"order_id":"42"}', 'status' => 'completed' }
    message = collect([function_call]).messages.first
    call = message.tool_calls.values.sole
    entry = message.ruby_llm_usage_entries.sole

    expect(call).to have_attributes(name: 'lookup_order', namespace: 'lookup_order', provider_call_id: 'order_1')
    expect(message).to equal(chat.messages.last)
    expect(entry.message).to equal(message)

    chat.with_tools(finder, target, defer: true).run_tools

    normalized = chat.messages.reverse.find(&:tool_call?)
    expect(chat.usage_entries.last).to equal(entry)
    expect(entry.message).to equal(normalized)
    expect(normalized.tool_calls.values.sole).to have_attributes(id: call.id, name: 'lookup_order', namespace: nil,
                                                                 provider_call_id: 'order_1')
    expect(finder.queries).to eq(['orders'])
    expect(chat.messages.last).to have_attributes(content: 'Order 42 shipped', tool_call_id: call.id,
                                                  provider_call_id: 'order_1')
    expect(chat.render[:input][-2]).to eq(function_call)
    expect(chat.render[:input].last).to include(type: 'function_call_output', call_id: 'order_1',
                                                output: 'Order 42 shipped')
  end

  it 'restores a flattened ordinary namespace before delivering a completion' do
    LookupOrder.namespace 'crm'
    chat = RubyLLM.chat(model: model_for(:openai, :tool_namespace), provider: :openai, protocol: :chat_completions)
                  .with_tools(target).ask_later('Find order 42')
    seen = []
    chat.after_message { |message| seen << message }
    body = {
      'model' => chat.model.id,
      'choices' => [{ 'finish_reason' => 'tool_calls', 'message' => {
        'role' => 'assistant', 'content' => nil,
        'tool_calls' => [{ 'id' => 'order_1', 'type' => 'function',
                           'function' => { 'name' => 'crm_lookup_order', 'arguments' => '{"order_id":"42"}' } }]
      } }],
      'usage' => { 'prompt_tokens' => 5, 'completion_tokens' => 2 }
    }
    protocol = RubyLLM::Providers::OpenAI.protocols.fetch(:chat_completions).new(chat.provider, chat.model)
    response = protocol.send(:parse_completion_body, body, raw: nil)

    message = chat.add_completion(response)
    call_id = message.tool_calls.values.sole.id

    expect(call_id).not_to be_nil
    expect(message.tool_calls.values.sole).to have_attributes(name: 'lookup_order', namespace: 'crm',
                                                              provider_call_id: 'order_1')
    expect(message).to equal(chat.messages.last)
    expect(seen.sole).to equal(message)
    expect(chat.usage_entries.sole.message).to equal(message)
    expect(message.tokens).to have_attributes(input: 5, output: 2)

    chat.run_tools

    expect(chat.messages.last).to have_attributes(content: 'Order 42 shipped', tool_call_id: call_id,
                                                  provider_call_id: 'order_1')
  end

  it 'waits for approval of the canonical finder before executing it' do
    FindTools.requires_approval
    message = collect([client_search_item]).messages.first
    chat.run_tools

    expect(chat).to be_waiting
    expect(finder.queries).to be_empty
    question = chat.pending_questions.find(&:approval?)
    expect(question).not_to be_nil
    expect(question.tool_call).to have_attributes(id: message.tool_calls.values.sole.id,
                                                  name: 'tool_search', provider_call_id: 'search_1')

    chat.approve(question)
    chat.run_tools

    expect(finder.queries).to eq(['orders'])
    expect(chat.pending_questions.select(&:approval?)).to be_empty
    expect(chat.messages.last.content).to eq('{"loaded":["lookup_order"]}')
  end
end
