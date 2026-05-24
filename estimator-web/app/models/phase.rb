class Phase
  include ActiveModel::Model
  include ActiveModel::Attributes

  attribute :name, :string
  attribute :duration_weeks, :integer
  attribute :cost_eur, :integer
  attribute :summary, :string
end
