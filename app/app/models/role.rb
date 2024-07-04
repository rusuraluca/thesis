class Role < ApplicationRecord
  # Represents a role that can be assigned to users.
  has_and_belongs_to_many :users, :join_table => :users_roles
  
  belongs_to :resource,
             :polymorphic => true,
             :optional => true


  # Validates that the resource_type is included in the list of resource types defined by Rolify, allowing nil values.
  validates :resource_type,
            :inclusion => { :in => Rolify.resource_types },
            :allow_nil => true

  scopify
end
