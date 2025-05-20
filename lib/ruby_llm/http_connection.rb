# frozen_string_literal: true

module RubyLLM
  # HTTP-based Connection class for Anthropic and Bedrock providers.
  # This implementation uses the HTTP gem instead of Faraday.
  class HttpConnection
    attr_reader :provider, :config

    def initialize(provider, config)
      @provider = provider
      @config = config

      ensure_configured!
    end

    def post(url, payload, &block)
      body = payload.is_a?(Hash) ? JSON.generate(payload, ascii_only: false) : payload
      headers = @provider.respond_to?(:headers) ? @provider.headers(@config) : {}
      
      options = {
        body: body,
        headers: headers,
        timeout: @config.request_timeout
      }
      
      # Apply any custom request options
      yield options if block_given?

      log_request('POST', url, options)
      
      # Add AWS auth headers if needed
      if @provider.respond_to?(:sign_request) && payload.is_a?(Hash)
        full_url = "#{@provider.api_base(@config)}/#{url}"
        signature = @provider.sign_request(full_url, config: @config, payload: payload)
        options[:headers].merge!(signature.headers)
      end
      
      # Create the request with retry support
      response = nil
      retry_with_backoff(@config.max_retries, @config.retry_interval, @config.retry_interval_randomness, @config.retry_backoff_factor) do
        # Set more appropriate timeouts for HTTP gem
        client = HTTP.headers(options[:headers])
                    .timeout(
                      connect: 60,     # Connection timeout
                      write: 60,       # Write timeout
                      read: options[:timeout] || 120  # Read timeout
                    )
        response = client.post(full_url(url), body: body)
        validate_response(response)
        response
      end
      
      handle_response(response)
    end

    def get(url, &block)
      headers = @provider.respond_to?(:headers) ? @provider.headers(@config) : {}
      
      options = {
        headers: headers,
        timeout: @config.request_timeout
      }
      
      # Apply any custom request options
      yield options if block_given?

      log_request('GET', url, options)
      
      # Create the request with retry support
      response = nil
      retry_with_backoff(@config.max_retries, @config.retry_interval, @config.retry_interval_randomness, @config.retry_backoff_factor) do
        # Set more appropriate timeouts for HTTP gem
        client = HTTP.headers(options[:headers])
                    .timeout(
                      connect: 60,     # Connection timeout
                      write: 60,       # Write timeout
                      read: options[:timeout] || 120  # Read timeout
                    )
        response = client.get(full_url(url))
        validate_response(response)
        response
      end
      
      handle_response(response)
    end

    # Method to handle streaming responses
    def post_stream(url, payload, &block)
      body = payload.is_a?(Hash) ? JSON.generate(payload, ascii_only: false) : payload
      headers = @provider.respond_to?(:headers) ? @provider.headers(@config) : {}
      
      # Add AWS signature headers if needed
      if @provider.respond_to?(:sign_request) && payload.is_a?(Hash)
        full_url = "#{@provider.api_base(@config)}/#{url}"
        signature = @provider.sign_request(full_url, config: @config, payload: payload)
        headers.merge!(signature.headers)
      end
      
      log_request('POST_STREAM', url, { body: body, headers: headers })
      
      # For AWS Bedrock streaming, we need to set accept header
      if @provider.is_a?(RubyLLM::Providers::Bedrock)
        headers['Accept'] = 'application/vnd.amazon.eventstream'
      end
      
      # Try the request with retries for non-streaming errors
      event_parser = EventStreamParser::Parser.new
      retry_with_backoff(@config.max_retries, @config.retry_interval, @config.retry_interval_randomness, @config.retry_backoff_factor) do
        # Set more appropriate timeouts for HTTP gem with longer streaming timeout
        client = HTTP.headers(headers)
                    .timeout(
                      connect: 60,     # Connection timeout
                      write: 60,       # Write timeout  
                      read: 300        # Longer read timeout for streaming
                    )
        
        # Stream the response directly to the user's block
        client.post(full_url(url), body: body).body.each do |chunk|
          # For Bedrock, we need to parse the chunk directly
          if @provider.is_a?(RubyLLM::Providers::Bedrock)
            if block_given?
              block.call(chunk)
            end
          else
            # For Anthropic and other providers, parse SSE events
            event_parser.feed(chunk) do |type, data|
              if block_given? && data != '[DONE]'
                data_obj = JSON.parse(data) rescue nil
                block.call(data_obj) if data_obj
              end
            end
          end
        end
      end
      
      # Return nil since we can't accumulate the stream
      nil
    end

    private

    def full_url(url)
      "#{provider.api_base(@config)}/#{url}".gsub(/([^:])\/\//, '\1/')
    end

    def handle_response(response)
      if response.status.success?
        response
      else
        handle_error_response(response)
      end
    end

    def validate_response(response)
      return if response.status.success?
      
      # Determine if we should retry based on response status
      retry_statuses = [429, 500, 502, 503, 504, 529]
      if retry_statuses.include?(response.status.code)
        raise RetryableHttpError.new(response)
      else
        handle_error_response(response)
      end
    end

    def handle_error_response(response)
      message = @provider&.parse_error(response)
      
      case response.status.code
      when 400
        raise BadRequestError.new(response, message || 'Invalid request - please check your input')
      when 401
        raise UnauthorizedError.new(response, message || 'Invalid API key - check your credentials')
      when 402
        raise PaymentRequiredError.new(response, message || 'Payment required - please top up your account')
      when 403
        raise ForbiddenError.new(response, message || 'Forbidden - you do not have permission to access this resource')
      when 429
        raise RateLimitError.new(response, message || 'Rate limit exceeded - please wait a moment')
      when 500
        raise ServerError.new(response, message || 'API server error - please try again')
      when 502..503
        raise ServiceUnavailableError.new(response, message || 'API server unavailable - please try again later')
      when 529
        raise OverloadedError.new(response, message || 'Service overloaded - please try again later')
      else
        raise Error.new(response, message || 'An unknown error occurred')
      end
    end

    def log_request(method, url, options)
      RubyLLM.logger.debug "#{method} #{full_url(url)}"
      RubyLLM.logger.debug "Headers: #{options[:headers]}" if options[:headers]
      
      # Log body but filter sensitive data
      if options[:body]
        body_log = options[:body].to_s
        body_log = body_log.gsub(%r{[A-Za-z0-9+/=]{100,}}, 'data":"[BASE64 DATA]"')
        body_log = body_log.gsub(/[-\d.e,\s]{100,}/, '[EMBEDDINGS ARRAY]')
        RubyLLM.logger.debug "Body: #{body_log}"
      end
    end

    def retry_with_backoff(max_retries, base_interval, randomness, backoff_factor, &block)
      retries = 0
      begin
        return yield
      rescue RetryableHttpError, HTTP::Error, Timeout::Error, Errno::ETIMEDOUT, 
             Errno::ECONNRESET, Errno::ECONNREFUSED, Errno::EHOSTUNREACH, 
             EOFError, SocketError, OpenSSL::SSL::SSLError => e
        
        if retries < max_retries
          retries += 1
          sleep_time = base_interval * (backoff_factor ** (retries - 1))
          sleep_time *= (1 + rand * randomness)
          RubyLLM.logger.debug "Retry #{retries}/#{max_retries} after #{sleep_time}s due to #{e.class}: #{e.message}"
          sleep sleep_time
          retry
        else
          raise e if e.is_a?(RetryableHttpError) && e.response
          raise ServerError.new(nil, "Maximum retries exceeded: #{e.message}")
        end
      end
    end

    def ensure_configured!
      return if @provider.configured?(@config)

      config_block = <<~RUBY
        RubyLLM.configure do |config|
          #{@provider.missing_configs(@config).map { |key| "config.#{key} = ENV['#{key.to_s.upcase}']" }.join("\n  ")}
        end
      RUBY

      raise ConfigurationError,
            "#{@provider.slug} provider is not configured. Add this to your initialization:\n\n#{config_block}"
    end
  end
  
  # Custom error class for retriable HTTP errors
  class RetryableHttpError < StandardError
    attr_reader :response
    
    def initialize(response)
      @response = response
      super("Retriable HTTP error: #{response.status.code}")
    end
  end
end