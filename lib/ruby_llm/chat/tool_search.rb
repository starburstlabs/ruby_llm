# frozen_string_literal: true

module RubyLLM
  class Chat
    # Deferred tool registration and the tool set rendered for the request.
    # A provider with tool search of its own gets every tool, the deferred
    # ones marked. Any other gets the tools that are not deferred, the
    # search tool, and the deferred tools the conversation has loaded with
    # it or called, in the order it first did, so every request repeats
    # the one before it and adds to its end.
    module ToolSearch # :nodoc:
      private

      def deferred_tool_names
        names = @tool_deferrals.filter_map { |name, deferred| name if deferred }
        mcp.each do |server|
          explicit = @mcp_deferrals[server.name.to_sym]
          MCP::Outages.tools(server).each do |tool|
            names << tool.name.to_sym if explicit.nil? ? server.defers?(tool) : explicit
          end
        end
        names
      end

      def request_tools
        deferred = deferred_tool_names
        return searched_request_tools(deferred) if searching_tools_here?(deferred)

        tools.to_h { |name, tool| [name, deferred.include?(name) ? Tool::Deferred.new(tool) : tool] }
      end

      def searched_request_tools(deferred)
        all = tools
        loaded = loaded_tool_names(deferred, all).select { |name| all.key?(name) }
        all.except(*deferred).merge(all.slice(*loaded))
      end

      def tool_search_tool
        return if deferred_tool_names.empty?

        registered = @tools.each_value.find(&:searches_tools?)
        return registered if registered
        return if @provider.tool_search?(@model, protocol: @protocol)

        @tool_search_tool ||= Tools::Search.new(-> { deferred_tools })
      end

      def searching_tools_here?(deferred)
        deferred.any? && !@provider.tool_search?(@model, protocol: @protocol)
      end

      def check_tool_search(name, tool)
        if tool.searches_tools? && (tool.name.to_s != 'tool_search' || !tool.namespace.nil?)
          raise ArgumentError, 'A tool declared with searches_tools must be named tool_search without a namespace'
        end

        existing = @tools[name]
        return unless existing && existing.searches_tools? != tool.searches_tools?

        raise ArgumentError, 'The tool_search name cannot be shared by a search tool and an ordinary tool'
      end

      def deferred_registration?(tool, defer)
        !tool.searches_tools? && (defer.nil? ? tool.deferred? : defer)
      end

      def loaded_tool_names(deferred, all)
        results = messages.select(&:tool_result?).to_h { |message| [message.tool_call_id, message.content] }
        calls = messages.select(&:tool_call?).flat_map { |message| message.tool_calls.to_a }
        calls.flat_map { |tool_call| names_loaded_by(tool_call, results, all) }.uniq & deferred
      end

      def names_loaded_by(tool_call, results, all)
        return [tool_call.qualified_name.to_sym] unless Tools::Search.search?(tool_call, all)

        Array(ToolSearchResults.from_content(results[tool_call.id])&.names).map(&:to_sym)
      end
    end
  end
end
