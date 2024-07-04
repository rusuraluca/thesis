class AddStatusToInquires < ActiveRecord::Migration[6.0]
  def change
    add_column :inquires, :status, :string, default: 'Not Verified'
  end
end
