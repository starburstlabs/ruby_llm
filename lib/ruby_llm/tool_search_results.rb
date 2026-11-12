# frozen_string_literal: true

module RubyLLM
  # The tools a search found, returned from #execute of a tool declared
  # with Tool.searches_tools. The chat loads the deferred tools it names,
  # and the model can call them from its next step on.
  #
  #   def execute(query:)
  #     RubyLLM::ToolSearchResults.new(*ToolIndex.nearest(query, limit: 5))
  #   end
  #
  # A name that is not one of the chat's deferred tools loads nothing.
  class ToolSearchResults
    include Support::Inspectable

    KEY = :loaded # :nodoc:

    # Returns the names of the tools to load, as the chat registers them:
    # a namespaced tool's namespace and name joined, as in
    # <tt>"crm_customer_lookup"</tt>.
    attr_reader :names

    # Returns a ToolSearchResults for +tools+, each a Tool class, a Tool,
    # or the name the chat registers it under. No arguments means nothing
    # was found. Raises ArgumentError for anything else.
    #
    #   RubyLLM::ToolSearchResults.new(LookupOrder, :crm_customer_lookup)
    #
    def initialize(*tools)
      @names = tools.map { |tool| qualified_name(tool) }.uniq
    end

    def self.from_content(content) # :nodoc:
      return unless content.to_s.start_with?('{')

      names = JSON.parse(content.to_s)[KEY.to_s]
      new(*names.map(&:to_s)) if names.is_a?(Array)
    rescue JSON::ParserError
      nil
    end

    def within(tools) # :nodoc:
      names.filter_map do |name|
        tool = tools[name.to_sym]
        [name.to_sym, tool] if tool.is_a?(Tool::Deferred)
      end.to_h
    end

    def to_h # :nodoc:
      { KEY => names }
    end

    def to_json(*) # :nodoc:
      JSON.generate(to_h, *)
    end

    private

    def inspect_attributes
      { names: }
    end

    def qualified_name(tool)
      case tool
      when String, Symbol then tool.to_s
      when Class then tool <= Tool ? [tool.namespace, tool.tool_name].compact.join('_') : invalid(tool)
      when Tool then tool.qualified_name
      else invalid(tool)
      end
    end

    def invalid(tool)
      raise ArgumentError, "ToolSearchResults takes Tool classes, Tools, or tool names, not #{tool.inspect}"
    end
  end
end
