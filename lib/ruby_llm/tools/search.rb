# frozen_string_literal: true

module RubyLLM
  module Tools
    module Search # :nodoc:
      module_function

      def handler(tools)
        handlers = tools.values.select(&:searches_tools?)
        raise ArgumentError, 'Register only one tool that declares searches_tools' if handlers.size > 1

        handlers.first
      end

      def resolve(results, tools)
        found, missing = results.names.partition do |name|
          tool = tools[name.to_sym]
          tool && !tool.searches_tools?
        end
        [found.map { |name| tools.fetch(name.to_sym) }, missing]
      end

      def validate(results, producer:, tools:)
        unless producer&.searches_tools?
          raise ArgumentError,
                'Only tools declaring searches_tools can return ToolSearchResults'
        end

        _, missing = resolve(results, tools)
        return if missing.empty?

        raise ArgumentError, "Tool search returned unavailable tools: #{missing.join(', ')}"
      end

      def replay(results, tools)
        found, missing = resolve(results, tools)
        RubyLLM.logger.warn("Omitting unavailable tool search results: #{missing.join(', ')}") if missing.any?
        found
      end
    end
  end
end
