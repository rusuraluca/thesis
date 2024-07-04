class DashboardsController < RestrictedAccessController
  # Controller for handling dashboard-related actions.
  # Inherits from RestrictedAccessController to ensure access control.

  def show
    # Displays the dashboard with paginated profiles.
    # @return [void]
    @profiles = Profile.page(params[:page]).per(3)
  end
end
