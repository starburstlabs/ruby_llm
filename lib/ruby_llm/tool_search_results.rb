# frozen_string_literal: true

module RubyLLM
  # Tools found by an application-defined search tool. Results refer to tools
  # already registered in the chat, using their model-visible names.
  #
  #   RubyLLM::ToolSearchResults.new(Weather, :lookup_order)
  class ToolSearchResults
    include Support::Inspectable

    KEY = 'tool_search_results' # :nodoc:

    # Returns the matching tool names in order, without duplicates.
    attr_reader :names

    # Returns results built from tool names, Tool classes, or Tool instances.
    # An empty list represents no matches.
    def initialize(*tools)
      @names = tools.map { |tool| normalize(tool) }.uniq
    end

    def self.from_content(content) # :nodoc:
      return unless content.is_a?(String) && content.lstrip.start_with?('{')

      parsed = JSON.parse(content)
      return unless parsed.is_a?(Hash)

      names = parsed[KEY]
      new(*names) if names.is_a?(Array) && names.all?(String)
    rescue JSON::ParserError
      nil
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

    def invalid_tool
      raise ArgumentError, 'Tool search results require tool names, Tool classes, or Tool instances'
    end

    def normalize(tool)
      case tool
      when String, Symbol then tool.to_s
      when Class
        tool <= Tool ? tool.tool_name.to_s : invalid_tool
      when Tool then tool.name.to_s
      else
        invalid_tool
      end
    end
  end
end
