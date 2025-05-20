# frozen_string_literal: true

module RubyLLM
  # Connection class for managing API connections to various providers.
  class Connection
    attr_reader :provider, :config

    def initialize(provider, config)
      @provider = provider
      @config = config
      ensure_configured!
    end

    def post(url, payload, &block)
      body = payload.is_a?(Hash) ? JSON.generate(payload, ascii_only: false) : payload
      headers = build_headers(&block)

      log_request('POST', url, headers, body)
      
      response = with_retry do
        HTTP.timeout(@config.request_timeout)
            .headers(headers)
            .post(build_url(url), body: body)
      end

      handle_response(response)
    end

    def get(url, &block)
      headers = build_headers(&block)

      log_request('GET', url, headers)

      response = with_retry do
        HTTP.timeout(@config.request_timeout)
            .headers(headers)
            .get(build_url(url))
      end

      handle_response(response)
    end

    private

    def build_headers
      headers = {}
      headers.merge!(@provider.headers(@config)) if @provider.respond_to?(:headers)
      
      # Let the block add additional headers if provided
      if block_given?
        req = OpenStruct.new(headers: headers)
        yield req
        headers = req.headers
      end
      
      headers
    end

    def build_url(path)
      base_url = @provider.api_base(@config)
      base_url = base_url.chomp('/') if base_url.end_with?('/')
      path = path.start_with?('/') ? path : "/#{path}"
      "#{base_url}#{path}"
    end

    def handle_response(response)
      log_response(response)
      
      # Create a response object similar to Faraday's
      response_obj = OpenStruct.new(
        status: response.status.code,
        body: parse_response_body(response),
        headers: response.headers.to_h,
        env: OpenStruct.new(url: response.uri.to_s)
      )

      # Check for errors
      check_for_errors(response_obj)
      
      response_obj
    end

    def parse_response_body(response)
      return {} if response.body.to_s.empty?
      
      content_type = response.content_type.mime_type rescue nil
      if content_type == 'application/json'
        begin
          JSON.parse(response.body.to_s)
        rescue JSON::ParserError
          response.body.to_s
        end
      else
        response.body.to_s
      end
    end

    def check_for_errors(response)
      status = response.status
      
      case status
      when 400...500
        case status
        when 401
          raise RubyLLM::AuthenticationError, "Authentication failed for #{@provider.slug}"
        when 403
          raise RubyLLM::PermissionError, "Permission denied for #{@provider.slug}"
        when 404
          raise RubyLLM::NotFoundError, "Resource not found at #{response.env.url}"
        when 429
          raise RubyLLM::RateLimitError, "Rate limit exceeded for #{@provider.slug}"
        else
          raise RubyLLM::ClientError, "Client error: #{status} for #{@provider.slug}"
        end
      when 500...600
        case status
        when 500, 520
          raise RubyLLM::ServerError, "Server error for #{@provider.slug}"
        when 503, 529
          raise RubyLLM::ServiceUnavailableError, "Service unavailable for #{@provider.slug}"
        when 502, 504
          raise RubyLLM::GatewayError, "Gateway error for #{@provider.slug}"
        else
          raise RubyLLM::ServerError, "Server error: #{status} for #{@provider.slug}"
        end
      end
    end

    def with_retry
      retries = 0
      begin
        yield
      rescue => e
        if retryable_error?(e) && retries < @config.max_retries
          retries += 1
          sleep calculate_retry_interval(retries)
          retry
        else
          raise e
        end
      end
    end

    def retryable_error?(error)
      retry_exceptions.any? { |klass| error.is_a?(klass) } ||
        (error.respond_to?(:status) && [429, 500, 502, 503, 504, 529].include?(error.status))
    end
    
    def retry_exceptions
      [
        Errno::ETIMEDOUT,
        Timeout::Error,
        HTTP::TimeoutError,
        HTTP::ConnectionError,
        RubyLLM::RateLimitError,
        RubyLLM::ServerError,
        RubyLLM::ServiceUnavailableError,
        RubyLLM::OverloadedError
      ]
    end

    def calculate_retry_interval(retries)
      interval = @config.retry_interval * (@config.retry_backoff_factor ** (retries - 1))
      randomness = interval * @config.retry_interval_randomness
      interval + rand(-randomness..randomness)
    end

    def log_request(method, url, headers, body = nil)
      return unless RubyLLM.logger.debug?

      redacted_headers = redact_sensitive_headers(headers)
      
      debug_message = "#{method} #{url}\n"
      debug_message += "Headers: #{redacted_headers.inspect}\n" if redacted_headers.any?
      
      if body && RubyLLM.logger.debug?
        if body.is_a?(String) && body.length > 1000
          debug_message += "Body: [LARGE REQUEST BODY - #{body.length} bytes]"
        else
          # Filter sensitive data
          filtered_body = body.to_s
          filtered_body = filtered_body.gsub(/[A-Za-z0-9+\/=]{100,}/, 'data":"[BASE64 DATA]"')
          filtered_body = filtered_body.gsub(/[-\d.e,\s]{100,}/, '[EMBEDDINGS ARRAY]')
          debug_message += "Body: #{filtered_body}"
        end
      end
      
      RubyLLM.logger.debug(debug_message)
    end

    def log_response(response)
      return unless RubyLLM.logger.debug?
      
      debug_message = "Response status: #{response.status.code}\n"
      
      if response.body && !response.body.to_s.empty?
        if response.body.to_s.length > 1000
          debug_message += "Body: [LARGE RESPONSE BODY - #{response.body.to_s.length} bytes]"
        else
          # Filter sensitive data
          body_str = response.body.to_s
          body_str = body_str.gsub(/[A-Za-z0-9+\/=]{100,}/, 'data":"[BASE64 DATA]"')
          body_str = body_str.gsub(/[-\d.e,\s]{100,}/, '[EMBEDDINGS ARRAY]')
          debug_message += "Body: #{body_str}"
        end
      end
      
      RubyLLM.logger.debug(debug_message)
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
end