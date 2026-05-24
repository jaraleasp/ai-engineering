require "faraday"

module EstimatorAi
  class Client
    Error              = Class.new(StandardError)
    InvalidRequest     = Class.new(Error)
    GuardrailViolation = Class.new(Error)
    ServerError        = Class.new(Error)

    GUARDRAIL_REASONS = %w[moderation prompt_injection pii].freeze

    def initialize(base_url: Rails.application.config.estimator_ai.base_url,
                   timeout:  Rails.application.config.estimator_ai.timeout)
      @conn = Faraday.new(url: base_url) do |f|
        f.request  :json
        f.response :json
        f.options.timeout = timeout
        f.adapter Faraday.default_adapter
      end
    end

    def estimate(request)
      raise ArgumentError, "request must be valid" unless request.valid?

      response = @conn.post("/api/v1/estimate", request.to_payload)

      case response.status
      when 200
        response.body
      when 400
        detail = extract_detail(response.body)
        reason = detail.is_a?(Hash) ? detail["reason"] : nil
        if GUARDRAIL_REASONS.include?(reason)
          message = detail.is_a?(Hash) ? detail["message"] || reason : reason
          raise GuardrailViolation, "Input rejected (#{reason}): #{message}"
        else
          raise InvalidRequest, detail.to_s
        end
      when 422
        raise InvalidRequest, extract_detail(response.body).to_s
      when 502
        raise ServerError, "Upstream LLM call failed"
      else
        raise ServerError, "unexpected status #{response.status}"
      end
    end

    private

    def extract_detail(body)
      return body unless body.is_a?(Hash)
      body["detail"] || body
    end
  end
end
