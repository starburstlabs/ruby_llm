# frozen_string_literal: true

module RubyLLM
  # Handles streaming responses from AI providers. Provides a unified way to process
  # chunked responses, accumulate content, and handle provider-specific streaming formats.
  # Each provider implements provider-specific parsing while sharing common stream handling
  # patterns.
  module Streaming
    module_function

    def stream_response(connection, payload, &block)
      # Default implementation for providers that don't implement specialized streaming
      # Individual providers can override this with their own implementation
      accumulator = StreamAccumulator.new

      connection.post stream_url, payload do |req|
        req.options.on_data = handle_stream do |chunk|
          accumulator.add chunk
          block.call chunk
        end
      end

      accumulator.to_message
    end

    def handle_stream(&block)
      to_json_stream do |data|
        block.call(build_chunk(data)) if data
      end
    end

    private

    def to_json_stream(&block)
      buffer = String.new
      parser = EventStreamParser::Parser.new

      proc do |chunk, _bytes, env|
        RubyLLM.logger.debug "Received chunk: #{chunk}"

        if error_chunk?(chunk)
          handle_error_chunk(chunk, env)
        elsif env&.status != 200
          handle_failed_response(chunk, buffer, env)
        else
          yield handle_sse(chunk, parser, env, &block)
        end
      end
    end

    def error_chunk?(chunk)
      chunk.start_with?('event: error')
    end

    def handle_error_chunk(chunk, env)
      error_data = chunk.split("\n")[1].delete_prefix('data: ')
      status, _message = parse_streaming_error(error_data)
      error_response = env.merge(body: JSON.parse(error_data), status: status)
      ErrorMiddleware.parse_error(provider: self, response: error_response)
    rescue JSON::ParserError => e
      RubyLLM.logger.debug "Failed to parse error chunk: #{e.message}"
    end

    def handle_failed_response(chunk, buffer, env)
      buffer << chunk
      error_data = JSON.parse(buffer)
      error_response = env.merge(body: error_data)
      ErrorMiddleware.parse_error(provider: self, response: error_response)
    rescue JSON::ParserError
      RubyLLM.logger.debug "Accumulating error chunk: #{chunk}"
    end

    def handle_sse(chunk, parser, env, &block)
      parser.feed(chunk) do |type, data|
        case type.to_sym
        when :error
          handle_error_event(data, env)
        else
          yield handle_data(data, &block) unless data == '[DONE]'
        end
      end
    end

    def handle_data(data)
      JSON.parse(data)
    rescue JSON::ParserError => e
      RubyLLM.logger.debug "Failed to parse data chunk: #{e.message}"
    end

    def handle_error_event(data, env)
      status, _message = parse_streaming_error(data)
      error_response = env.merge(body: JSON.parse(data), status: status)
      ErrorMiddleware.parse_error(provider: self, response: error_response)
    rescue JSON::ParserError => e
      RubyLLM.logger.debug "Failed to parse error event: #{e.message}"
    end
  end
end
