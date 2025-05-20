# frozen_string_literal: true

module RubyLLM
  # Handles streaming responses from AI providers. Provides a unified way to process
  # chunked responses, accumulate content, and handle provider-specific streaming formats.
  # Each provider implements provider-specific parsing while sharing common stream handling
  # patterns.
  module Streaming
    module_function

    def stream_response(connection, payload, &block)
      accumulator = StreamAccumulator.new
      
      # Build URL
      base_url = connection.provider.api_base(connection.config)
      base_url = base_url.chomp('/') if base_url.end_with?('/')
      url = stream_url.start_with?('/') ? stream_url : "/#{stream_url}"
      full_url = "#{base_url}#{url}"
      
      # Prepare headers
      headers = {}
      headers.merge!(connection.provider.headers(connection.config)) if connection.provider.respond_to?(:headers)
      
      # Prepare body
      body = payload.is_a?(Hash) ? JSON.generate(payload, ascii_only: false) : payload
      
      # Log request
      RubyLLM.logger.debug "POST #{full_url}"
      RubyLLM.logger.debug "Headers: #{redact_sensitive_headers(headers).inspect}"
      
      # Stream response using HTTP.rb
      parser = EventStreamParser::Parser.new
      buffer = String.new
      
      begin
        HTTP.timeout(connection.config.request_timeout)
            .headers(headers)
            .post(full_url, body: body) do |response|
              if response.status.success?
                response.body.each do |chunk|
                  RubyLLM.logger.debug "Received chunk: #{chunk}"
                  
                  if error_chunk?(chunk)
                    handle_error_chunk(chunk)
                  else
                    process_sse_chunk(chunk, parser, accumulator, &block)
                  end
                end
              else
                # Handle error response
                response.body.each do |chunk|
                  buffer << chunk
                end
                
                begin
                  error_data = JSON.parse(buffer)
                  error_response = OpenStruct.new(
                    status: response.status.code,
                    body: error_data,
                    headers: response.headers.to_h,
                    env: OpenStruct.new(url: full_url)
                  )
                  RubyLLM::ErrorHandler.parse_error(provider: connection.provider, response: error_response)
                rescue JSON::ParserError
                  raise RubyLLM::Error.new(nil, "Failed to parse error response: #{buffer}")
                end
              end
            end
      rescue HTTP::Error => e
        raise RubyLLM::Error.new(nil, "HTTP request failed: #{e.message}")
      end
      
      accumulator.to_message
    end
    
    private
    
    def process_sse_chunk(chunk, parser, accumulator, &block)
      parser.feed(chunk) do |type, data|
        case type.to_sym
        when :error
          handle_error_event(data)
        else
          unless data == '[DONE]'
            begin
              parsed_data = JSON.parse(data)
              message_chunk = build_chunk(parsed_data)
              accumulator.add(message_chunk)
              block.call(message_chunk)
            rescue JSON::ParserError => e
              RubyLLM.logger.debug "Failed to parse data chunk: #{e.message}"
            end
          end
        end
      end
    end

    def error_chunk?(chunk)
      chunk.start_with?('event: error')
    end

    def handle_error_chunk(chunk)
      error_data = chunk.split("\n")[1].delete_prefix('data: ')
      begin
        error_json = JSON.parse(error_data)
        raise RubyLLM::Error.new(nil, error_json['error'] || error_json['message'] || "Error in streaming response")
      rescue JSON::ParserError => e
        RubyLLM.logger.debug "Failed to parse error chunk: #{e.message}"
        raise RubyLLM::Error.new(nil, "Error in streaming response: #{chunk}")
      end
    end

    def handle_error_event(data)
      begin
        error_json = JSON.parse(data)
        error_message = error_json['error'] || error_json['message'] || "Error in streaming response"
        raise RubyLLM::Error.new(nil, error_message)
      rescue JSON::ParserError => e
        RubyLLM.logger.debug "Failed to parse error event: #{e.message}"
        raise RubyLLM::Error.new(nil, "Error in streaming event: #{data}")
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