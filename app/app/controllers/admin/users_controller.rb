class Admin::UsersController < ApplicationController
  # Controller for handling administrative actions on users.

  before_action :authenticate_user!
  before_action :require_admin

  def index
    # Displays a list of users, paginated.
    # @return [void]
    @users = User.page(params[:page]).per(2)
  end

  def assign_role
    # Renders the form for assigning roles to a specific user.
    # @return [void]
    @user = User.find(params[:id])
    @roles = Role.all
  end

  def update_role
    # Updates the roles of a specific user.
    # @return [void]
    user = User.find(params[:id])
    user.roles.delete_all
    params[:roles].each do |role|
      user.add_role(role)
    end if params[:roles].present?
    redirect_to admin_users_path, notice: 'Roles updated successfully.'
  end

  def destroy
    # Deletes a specific user.
    # @return [void]
    @user = User.find(params[:id])
    @user.destroy
    redirect_to admin_users_path, notice: 'User was successfully deleted.'
  end

  private

  def require_admin
    # Ensures that the current user is an admin.
    # Redirects to the root path with an alert if the user is not an admin.
    # @return [void]
    unless current_user.has_role?(:admin)
      redirect_to root_path, alert: 'Unauthorized access!'
    end
  end
end
