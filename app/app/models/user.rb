class User < ApplicationRecord
  # Represents a user of the application with authentication and role management.
  rolify
  # Include default Devise modules for authentication and account management.
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable,
         :confirmable, :lockable, :timeoutable, :trackable

  # Validates presence and uniqueness of email.
  validates :email, presence: true, uniqueness: true

  # Assigns a default role to the user after creation if no roles are assigned.
  after_create :assign_default_role

  private

  def assign_default_role
    # Adds the 'regular' role to the user if no roles are present.
    self.add_role(:regular) if self.roles.blank?
  end
end
