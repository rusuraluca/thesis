class Inquire < ApplicationRecord
  # Represents an inquiry made by a user, associated with a profile and user.
  belongs_to :profile
  belongs_to :user

  has_many_attached :images

  # Defines the possible statuses for an inquiry.
  enum status: { not_verified: 'Not Verified', solved: 'Solved', not_solved: 'Not Solved' }

  # Validates presence of date_taken, city_taken, and country_taken fields.
  validates :date_taken, :city_taken, :country_taken, presence: true
end
