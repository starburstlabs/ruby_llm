# frozen_string_literal: true

module RubyLLM
  module Providers
    module Bedrock
      # Streaming implementation for the AWS Bedrock API using converse_stream API
      # This provides real-time streaming of AI model responses
      module Streaming
        module_function

        # Process a streaming chat completion request
        # @param connection [RubyLLM::Connection] The connection object
        # @param payload [Hash] The request payload to send
        # @param block [Proc] The block to call with each streaming chunk
        # @return [RubyLLM::Message] The complete message after streaming finishes
        def stream_response(connection, payload, &block)
          client = bedrock_runtime_client(connection.config)
          
          # Create accumulators for content and tool calls
          stream_data = {
            content: String.new,
            model_id: @model_id,
            input_tokens: nil,
            output_tokens: nil,
            tool_calls: {},
            current_tool_call_id: nil
          }
          
          # Debug information
          RubyLLM.logger.debug { "Original payload: #{payload.inspect}" }
          
          # Prepare the request for converse_stream API
          converse_request = build_converse_request(payload)
          
          # Debug the formatted request for converse_stream
          RubyLLM.logger.debug { "Converse request: #{converse_request.inspect}" }
          
          begin
            # Process the streaming response with callbacks
            client.converse_stream(converse_request) do |stream|
              # Handle content deltas (the main text output)
              stream.on_content_block_delta_event do |event|
                process_content_delta(event, stream_data, &block)
              end
              
              # Handle message start events
              stream.on_message_start_event do |event|
                process_message_start(event, stream_data)
              end
              
              # Handle message stop events
              stream.on_message_stop_event do |event|
                process_message_stop(event, stream_data, &block)
              end
              
              # Handle content block start events
              stream.on_content_block_start_event do |event|
                process_content_block_start(event, stream_data, &block)
              end
              
              # Handle errors during streaming
              stream.on_error_event do |event|
                RubyLLM.logger.error "Error in streaming: #{event.message}"
              end

              # Handle content block stop events
              stream.on_content_block_stop_event do |event|
                process_content_block_stop(event, stream_data, &block)
              end
            end

            # Build the final message from accumulated data
            build_message(stream_data)
          rescue Aws::Errors::ServiceError => e
            handle_aws_error(e)
          rescue => e
            RubyLLM.logger.error "Error in streaming: #{e.message}\n#{e.backtrace.join("\n")}"
            raise StandardError.new(
              "Failed to process Bedrock streaming response: #{e.message}"
            )
          end
        end
        
        # Process content delta event
        # @param event [Aws::BedrockRuntime::Types::ContentBlockDeltaEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_content_delta(event, stream_data, &block)
          return unless event.delta
          
          # Simplified version that ignores tool usage for now to ensure basic streaming works
          # Handle text content only
          return unless event.delta.respond_to?(:text) && event.delta.text
          
          text = event.delta.text
          return if text.empty?
          
          # Log the text chunk for debugging with timestamp
          RubyLLM.logger.debug { "Bedrock streaming text chunk at #{Time.now.strftime('%H:%M:%S.%L')}: #{text.inspect}" }
          
          # Update the accumulated content
          stream_data[:content] += text
          
          # Create a chunk for the streaming client and yield it immediately
          if block_given?
            stream_chunk = { "content" => text }
            block.call(stream_chunk)
            
            # Force flush to ensure real-time display of content
            $stdout.flush if $stdout.tty?
          end
        end
        
        # Process message start event
        # @param event [Aws::BedrockRuntime::Types::MessageStartEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        def process_message_start(event, stream_data)
          # Extract token counts if available
          if event.respond_to?(:metrics) && event.metrics
            stream_data[:input_tokens] = event.metrics.input_tokens if event.metrics.respond_to?(:input_tokens)
          end
          
          # Extract model information if available
          if event.respond_to?(:model) && event.model
            stream_data[:model_id] = event.model
          end
        end
        
        # Process message stop event
        # @param event [Aws::BedrockRuntime::Types::MessageStopEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_message_stop(event, stream_data, &block)
          # Extract token counts if available
          if event.respond_to?(:metrics) && event.metrics
            stream_data[:output_tokens] = event.metrics.output_tokens if event.metrics.respond_to?(:output_tokens)
          end
          
          # Signal completion to the client
          if block_given?
            completion_chunk = { "completion" => true }
            block.call(completion_chunk)
          end
        end
        
        # Process content block start event
        # @param event [Aws::BedrockRuntime::Types::ContentBlockStartEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_content_block_start(event, stream_data, &block)
          return unless event.content_block
          
          # Simplified version that focuses on text content only
          return unless event.content_block.respond_to?(:text) && event.content_block.text
          
          text = event.content_block.text
          return if text.empty?
          
          # Update the accumulated content
          stream_data[:content] += text
          
          # Create a chunk for the streaming client and yield it immediately
          if block_given?
            stream_chunk = { "content" => text }
            block.call(stream_chunk)
            
            # Force flush to ensure real-time display of content
            $stdout.flush if $stdout.tty?
          end
        end

        # Process content block stop event
        # @param event [Aws::BedrockRuntime::Types::ContentBlockStopEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_content_block_stop(event, stream_data, &block)
          # Currently no specific processing needed for content block stop
          # This method exists for completeness and future extensions
        end

        # Process tool use event - direct tool events
        # @param event [Aws::BedrockRuntime::Types::ToolUseEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_tool_use(event, stream_data, &block)
          return unless event.tool_use
          
          tool_use = event.tool_use
          tool_id = tool_use.id || SecureRandom.uuid
          stream_data[:current_tool_call_id] = tool_id
          
          # Extract tool name and arguments
          name = tool_use.name.to_s
          args = tool_use.input || {}
          
          # Initialize or update the tool call in our stream data
          stream_data[:tool_calls][tool_id] = ToolCall.new(
            id: tool_id,
            name: name,
            arguments: args
          )
          
          # Notify client about tool call if streaming
          if block_given?
            # Create a chunk for tool call
            stream_chunk = {
              "tool_call" => {
                "id" => tool_id,
                "name" => name,
                "arguments" => args
              }
            }
            block.call(stream_chunk)
          end
        end
        
        # Process tool use start from content_block_start
        # @param event [Aws::BedrockRuntime::Types::ContentBlockStartEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_tool_use_start(event, stream_data, &block)
          return unless event.content_block && event.content_block.respond_to?(:tool_use) && event.content_block.tool_use
          
          tool_use = event.content_block.tool_use
          tool_id = tool_use.id || SecureRandom.uuid
          stream_data[:current_tool_call_id] = tool_id
          
          # Extract tool information
          name = tool_use.respond_to?(:name) ? tool_use.name.to_s : "unknown_tool"
          args = tool_use.respond_to?(:input) ? (tool_use.input || {}) : {}
          
          # Initialize the tool call
          stream_data[:tool_calls][tool_id] = ToolCall.new(
            id: tool_id,
            name: name,
            arguments: args
          )
          
          # Notify client about tool call if streaming
          if block_given?
            # Create a chunk for tool call
            stream_chunk = {
              "tool_call" => {
                "id" => tool_id,
                "name" => name,
                "arguments" => args
              }
            }
            block.call(stream_chunk)
          end
        end
        
        # Process tool use delta from content_block_delta
        # @param event [Aws::BedrockRuntime::Types::ContentBlockDeltaEvent] The event to process
        # @param stream_data [Hash] Accumulator for stream data
        # @param block [Proc] The block to call with each streaming chunk
        def process_tool_use_delta(event, stream_data, &block)
          return unless event.delta && event.delta.respond_to?(:tool_use) && event.delta.tool_use
          
          tool_delta = event.delta.tool_use
          
          # If we have a tool ID, use it; otherwise use the current one
          if tool_delta.respond_to?(:id) && tool_delta.id
            tool_id = tool_delta.id
            stream_data[:current_tool_call_id] = tool_id
            
            # Create new tool call if it doesn't exist
            if !stream_data[:tool_calls][tool_id]
              name = tool_delta.respond_to?(:name) ? tool_delta.name.to_s : "unknown_tool"
              args = tool_delta.respond_to?(:input) ? (tool_delta.input || {}) : {}
              
              stream_data[:tool_calls][tool_id] = ToolCall.new(
                id: tool_id,
                name: name,
                arguments: args
              )
              
              # Notify client about new tool call
              if block_given?
                stream_chunk = {
                  "tool_call" => {
                    "id" => tool_id,
                    "name" => name,
                    "arguments" => args
                  }
                }
                block.call(stream_chunk)
              end
            end
          elsif stream_data[:current_tool_call_id]
            # Update existing tool call with delta information
            tool_id = stream_data[:current_tool_call_id]
            current_tool = stream_data[:tool_calls][tool_id]
            
            if current_tool && tool_delta.respond_to?(:input) && tool_delta.input
              # Update arguments based on their type
              if current_tool.arguments.is_a?(String) && tool_delta.input.is_a?(String)
                current_tool.arguments += tool_delta.input
              elsif current_tool.arguments.is_a?(Hash) && tool_delta.input.is_a?(Hash)
                current_tool.arguments.merge!(tool_delta.input)
              end
              
              # Notify client about tool call update
              if block_given?
                stream_chunk = {
                  "tool_call_update" => {
                    "id" => tool_id,
                    "arguments" => tool_delta.input
                  }
                }
                block.call(stream_chunk)
              end
            end
          end
        end
        
        # Build the final message from accumulated stream data
        # @param stream_data [Hash] The accumulated stream data
        # @return [RubyLLM::Message] The complete message
        def build_message(stream_data)
          # Convert tool calls to the expected format
          tool_calls = {}
          
          stream_data[:tool_calls].each do |id, tool_call|
            # Parse JSON arguments if they're in string form
            arguments = if tool_call.arguments.is_a?(String) && !tool_call.arguments.empty?
                        begin
                          JSON.parse(tool_call.arguments)
                        rescue JSON::ParserError
                          # If parsing fails, keep the string format
                          tool_call.arguments
                        end
                      elsif tool_call.arguments.is_a?(Hash)
                        # Ensure all symbol keys are converted to strings for consistency
                        tool_call.arguments.transform_keys(&:to_s)
                      else
                        tool_call.arguments
                      end
            
            tool_calls[id] = ToolCall.new(
              id: id,
              name: tool_call.name,
              arguments: arguments
            )
          end
          
          # Build and return the complete message
          Message.new(
            role: :assistant,
            content: stream_data[:content].empty? ? nil : stream_data[:content],
            model_id: stream_data[:model_id],
            tool_calls: tool_calls.empty? ? nil : tool_calls,
            input_tokens: stream_data[:input_tokens],
            output_tokens: stream_data[:output_tokens]
          )
        end

        private
        
        # Build a request for the converse_stream API
        # @param payload [Hash] The original payload for invoke_model
        # @return [Hash] The request for converse_stream
        def build_converse_request(payload)
          request = {
            model_id: @model_id,
            inference_config: {}
          }
          
          # Extract anthropic_version if present
          anthropic_version = payload[:anthropic_version] || payload['anthropic_version']
          
          # Handle max_tokens
          max_tokens = payload[:max_tokens] || payload['max_tokens']
          request[:inference_config][:max_tokens] = max_tokens if max_tokens
          
          # Handle temperature if present
          temperature = payload[:temperature] || payload['temperature']
          request[:inference_config][:temperature] = temperature if temperature
          
          # Handle top_p if present
          top_p = payload[:top_p] || payload['top_p']
          request[:inference_config][:top_p] = top_p if top_p
          
          # Handle messages array - ensure content format matches AWS SDK requirements
          if payload[:messages] || payload['messages']
            messages = deep_symbolize_keys(payload[:messages] || payload['messages'])
            
            # Format messages to match AWS SDK requirements (no 'type' field)
            properly_formatted_messages = messages.map do |message|
              new_message = message.dup
              
              # Convert string content to proper format array with text field only
              if new_message[:content].is_a?(String)
                new_message[:content] = [{ text: new_message[:content] }]
              elsif new_message[:content].is_a?(Array)
                # Already array format, make sure each item has proper format
                new_message[:content] = new_message[:content].map do |content_item|
                  if content_item.is_a?(String)
                    { text: content_item }
                  else
                    # It's a hash, remove type if present (AWS SDK handles types internally)
                    content_item = deep_symbolize_keys(content_item)
                    
                    # AWS SDK specifically requires no explicit type field
                    if content_item.key?(:type)
                      content_item = content_item.dup
                      content_item.delete(:type)
                    end
                    
                    # Make sure all keys and values are properly formatted for AWS SDK
                    content_item.transform_values do |v|
                      v.is_a?(Symbol) ? v.to_s : v
                    end
                  end
                end
              end
              
              new_message
            end
            
            request[:messages] = properly_formatted_messages
          end
          
          # Add system prompt if provided
          if payload[:system] || payload['system']
            system_content = payload[:system] || payload['system']
            request[:messages] ||= []
            
            # Check if we already have a system message
            system_exists = request[:messages].any? { |m| m[:role] == 'system' || m['role'] == 'system' }
            
            # Add system message if it doesn't exist
            unless system_exists
              request[:messages].unshift({
                role: 'system',
                content: [{ text: system_content }]
              })
            end
          end

          # Skip tool configurations for now to ensure basic functionality works
          # We'll implement tool support later
          
          request
        end
        
        # Helper to deep symbolize keys in a hash or array of hashes
        # @param obj [Hash, Array] The object to symbolize keys in
        # @return [Hash, Array] The object with symbolized keys
        def deep_symbolize_keys(obj)
          case obj
          when Hash
            obj.each_with_object({}) do |(key, value), result|
              result[key.to_sym] = deep_symbolize_keys(value)
            end
          when Array
            obj.map { |item| deep_symbolize_keys(item) }
          else
            obj
          end
        end
      end
    end
  end
end