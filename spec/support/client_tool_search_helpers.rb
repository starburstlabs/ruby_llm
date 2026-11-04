# frozen_string_literal: true

RSpec.shared_context 'with client tool search' do
  include_context 'with configured RubyLLM'

  before do
    stub_const('FindTools', Class.new(RubyLLM::Tool) do
      description 'Finds tools for a task'
      searches_tools
      attr_accessor :result
      attr_reader :queries

      def initialize
        super
        @result = RubyLLM::ToolSearchResults.new(:lookup_order)
        @queries = []
      end

      def execute(query:)
        @queries << query
        result
      end
    end)
    stub_const('LookupOrder', Class.new(RubyLLM::Tool) do
      description 'Looks up an order'
      def execute(order_id:) = "Order #{order_id} shipped"
    end)
  end

  let(:finder) { FindTools.new }
  let(:target) { LookupOrder.new }

  def client_search_item(call_id: 'search_1', arguments: { 'query' => 'orders' })
    { 'type' => 'tool_search_call', 'execution' => 'client', 'id' => 'item_1', 'call_id' => call_id,
      'arguments' => arguments, 'status' => 'completed' }
  end

  def pending_search_message(raw: nil, name: finder.name, call_id: 'search_1')
    call = RubyLLM::ToolCall.new(id: call_id, name:, arguments: { 'query' => 'orders' })
    RubyLLM::Message.new(role: :assistant, content: nil, tool_calls: { call.id => call }, raw_content: raw)
  end

  def search_result_message(result = RubyLLM::ToolSearchResults.new(:lookup_order), call_id: 'search_1')
    content, = RubyLLM::Tool.split_result(result)
    RubyLLM::Message.new(role: :tool, content:, tool_call_id: call_id)
  end

  def responses_body(output, model: 'gpt-5.4')
    { 'output' => output, 'model' => model, 'status' => 'completed', 'usage' => {} }
  end

  def json_response(body)
    { status: 200, headers: { 'Content-Type' => 'application/json' }, body: JSON.generate(body) }
  end
end
