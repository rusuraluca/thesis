class PagesController < ApplicationController
  before_action :check_signed_in

  def home
    # Renders the home page.
  end

  private

  def check_signed_in
    # Checks if the user is signed in.
    # Redirects to the dashboard if the user is signed in.
    redirect_to(dashboard_path) if signed_in?
  end
end
