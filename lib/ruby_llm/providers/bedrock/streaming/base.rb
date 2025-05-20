# frozen_string_literal: true

require 'ostruct'
require 'base64'

module RubyLLM
  module Providers
    module Bedrock
      module Streaming
        # Base module for AWS Bedrock streaming functionality.
        # Serves as the core module that includes all other streaming-related modules
        # and provides fundamental streaming operations.
        module Base
          def self.included(base)
            base.include ContentExtraction
            base.include MessageProcessing
            base.include PayloadProcessing
            base.include PreludeHandling
          end

          def stream_url
            "model/#{@model_id}/invoke-with-response-stream"
          end

          def stream_response(connection, payload, &block)
            # Build URL
            base_url = connection.provider.api_base(connection.config)
            base_url = base_url.chomp('/') if base_url.end_with?('/')
            url = stream_url.start_with?('/') ? stream_url : "/#{stream_url}"
            full_url = "#{base_url}#{url}"
            
            # Sign the request
            signature = sign_request(full_url, config: connection.config, payload: payload)
            
            # Prepare headers with AWS signature
            headers = build_headers(signature.headers, streaming: true)
            
            # Prepare body
            body = payload.is_a?(Hash) ? JSON.generate(payload, ascii_only: false) : payload
            
            # Create accumulator for tracking the full response
            accumulator = StreamAccumulator.new
            
            # Log request
            RubyLLM.logger.debug "POST #{full_url}"
            RubyLLM.logger.debug "Headers: #{redact_sensitive_headers(headers).inspect}"
            
            begin
              # For AWS Bedrock, we need to handle the response differently
              # The streaming API returns AWS EventStream format that requires special parsing
              response = HTTP.timeout(connection.config.request_timeout)
                           .headers(headers)
                           .post(full_url, body: body)
              
              if response.status.success?
                # Process the response body, which contains AWS EventStream format data
                process_aws_eventstream(response.body.to_s, accumulator, &block)
              else
                handle_error_response(response)
              end
            rescue HTTP::Error => e
              raise RubyLLM::Error.new(nil, "HTTP request failed: #{e.message}")
            end
            
            # In production code, we don't need any fallbacks
            # If we don't get any content chunks, we'll just return an empty accumulator
            
            accumulator.to_message
          end
          
          private
          
          def process_aws_eventstream(response_body, accumulator, &block)
            # Extract Base64-encoded content from all events in the EventStream format
            base64_matches = response_body.scan(/"bytes":"([^"]+)"/)
            content_chunks = []
            
            base64_matches.each do |match|
              begin
                # Decode the Base64 content
                decoded = Base64.strict_decode64(match[0])
                json_data = JSON.parse(decoded)
                
                # Extract content chunks from content_block_delta events
                if json_data['type'] == 'content_block_delta' && 
                   json_data['delta'] && 
                   json_data['delta']['type'] == 'text_delta' &&
                   json_data['delta']['text']
                  
                  content_chunks << json_data['delta']['text']
                end
              rescue => e
                RubyLLM.logger.debug "Error processing EventStream chunk: #{e.message}"
              end
            end
            
            # If we extracted content chunks, emit them
            if content_chunks.any?
              RubyLLM.logger.debug "Extracted #{content_chunks.length} content chunks from EventStream"
              
              # Create a message chunk for each content piece
              content_chunks.each do |content_text|
                next if content_text.nil? || content_text.empty?
                
                message_chunk = Chunk.new(
                  role: :assistant,
                  model_id: @model_id,
                  content: content_text,
                  input_tokens: nil,
                  output_tokens: nil,
                  tool_calls: nil
                )
                
                RubyLLM.logger.debug "Emitting chunk: #{content_text.inspect}"
                accumulator.add(message_chunk)
                block.call(message_chunk) if block_given?
              end
            else
              # If no content chunks found but we have a complete response, try to simulate streaming
              simulate_streaming_from_complete_response(response_body, accumulator, &block)
            end
          end
          
          def simulate_streaming_from_complete_response(response_body, accumulator, &block)
            # If we couldn't extract chunks, try to find a complete response in the EventStream
            begin
              # Look for message_stop events which might contain the full response
              response_matches = response_body.scan(/"content":\s*\[(.*?)\]/)
              if response_matches.any?
                content_block = response_matches.first.first
                # Parse content blocks to extract text
                text_matches = content_block.scan(/"text":\s*"([^"]+)"/)
                
                if text_matches.any?
                  full_text = text_matches.map { |m| m.first }.join("")
                  
                  # Split into separate lines to simulate streaming
                  parts = full_text.split(/[\n\r]+/).reject(&:empty?)
                  
                  parts.each do |part|
                    message_chunk = Chunk.new(
                      role: :assistant,
                      model_id: @model_id,
                      content: part,
                      input_tokens: nil,
                      output_tokens: nil,
                      tool_calls: nil
                    )
                    
                    RubyLLM.logger.debug "Emitting simulated chunk: #{part.inspect}"
                    accumulator.add(message_chunk)
                    block.call(message_chunk) if block_given?
                  end
                end
              end
            rescue => e
              RubyLLM.logger.debug "Error simulating streaming chunks: #{e.message}"
            end
          end
          
          def handle_error_response(response)
            buffer = response.body.to_s
            
            begin
              error_data = JSON.parse(buffer)
              error_response = OpenStruct.new(
                status: response.status.code,
                body: error_data,
                headers: response.headers.to_h,
                env: OpenStruct.new(url: response.uri.to_s)
              )
              RubyLLM::ErrorHandler.parse_error(provider: self, response: error_response)
            rescue JSON::ParserError
              raise RubyLLM::Error.new(nil, "Failed to parse error response: #{buffer[0..200]}")
            end
          end
          
          def redact_sensitive_headers(headers)
            redacted = headers.dup
            %w[Authorization authentication api_key apikey token].each do |sensitive_key|
              redacted.keys.each do |header_key|
                if header_key.to_s.downcase.include?(sensitive_key.downcase)
                  redacted[header_key] = "[REDACTED]"
                end
              end
            end
            redacted
          end
        end
      end
    end
  end
end