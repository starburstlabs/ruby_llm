# frozen_string_literal: true

module RubyLLM
  module Protocols
    class Responses
      # Tools methods of the OpenAI Responses API. Function definitions are
      # flat rather than nested under a `function` key.
      module Tools
        module_function

        NATIVE_TOOL_SEARCH = { type: 'tool_search' }.freeze

        def tool_for(tool)
          if tool.searches_tools?
            return { type: 'tool_search', execution: 'client', description: tool.description,
                     parameters: ChatCompletions::Tools.parameters_schema_for(tool) }
          end

          definition = {
            type: 'function',
            name: tool.name,
            description: tool.description,
            parameters: ChatCompletions::Tools.parameters_schema_for(tool),
            strict: false
          }
          definition[:defer_loading] = true if deferred?(tool)

          return definition if tool.provider_options.empty?

          RubyLLM::Support::Utils.deep_merge(definition, tool.provider_options)
        end

        def deferred?(tool)
          tool.is_a?(RubyLLM::Tool::Deferred)
        end

        # The API requires a search tool next to deferred tools, and rejects
        # search items in the input of a request that declares none.
        def apply_tool_search(payload)
          tools = Array(payload[:tools])
          return payload.merge(input: without_hosted_search_items(payload[:input])) if client_search_only?(tools)
          return payload if tools.any? { |tool| search_tool?(tool) }
          return payload.merge(tools: tools + [NATIVE_TOOL_SEARCH.dup]) if defers_loading?(tools)

          payload.merge(input: payload[:input].reject do |item|
            Chat::TOOL_SEARCH_ITEM_TYPES.include?(item[:type] || item['type'])
          end)
        end

        def client_search_only?(tools)
          searches = tools.select { |tool| search_tool?(tool) }
          searches.any? && searches.all? { |tool| (tool[:execution] || tool['execution']) == 'client' }
        end

        def without_hosted_search_items(input)
          input.reject do |item|
            type = item[:type] || item['type']
            Chat::TOOL_SEARCH_ITEM_TYPES.include?(type) && (item[:execution] || item['execution']) != 'client'
          end
        end

        def defers_loading?(tools)
          tools.any? { |tool| tool[:defer_loading] || tool['defer_loading'] }
        end

        def search_tool?(tool)
          (tool[:type] || tool['type']) == NATIVE_TOOL_SEARCH[:type]
        end

        def build_tool_choice(tool_choice)
          if @search_tool && tool_choice.to_s == @search_tool.name
            raise ArgumentError, 'Responses cannot force a named client tool search; use choice: :auto or :required'
          end

          case tool_choice
          when :auto, :none, :required
            tool_choice
          else
            {
              type: 'function',
              name: tool_choice
            }
          end
        end
      end
    end
  end
end
