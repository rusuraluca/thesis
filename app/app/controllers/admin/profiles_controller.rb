class Admin::ProfilesController < ApplicationController
  # Controller for handling administrative actions on profiles.

  before_action :set_profile, only: [:show, :edit, :update, :destroy]
  def show
    # Displays a specific profile.
  end

  def new
    # Initializes a new profile.
    # @return [void]
    @profile = Profile.new
  end

  def edit
    # Renders the edit form for a specific profile.
    # @return [void]
  end

  def create
    # Creates a new profile with the provided parameters.
    # @return [void]
    @profile = Profile.new(profile_params)
    if @profile.save
      redirect_to [:admin, @profile], notice: 'Profile was successfully created.'
    else
      render :new, alert: 'Profile could not be created.'
    end
  end

  def update
    # Updates a specific profile with the provided parameters.
    # @return [void]
    if @profile.update(profile_params)
      redirect_to [:admin, @profile], notice: 'Profile was successfully updated.'
    else
      render :edit, alert: 'Profile could not be updated.'
    end
  end

  def destroy
    # Deletes a specific profile.
    # @return [void]
    @profile.destroy
    redirect_to dashboard_path, notice: 'Profile was successfully destroyed.'
  end

  private
  def set_profile
    # Sets the profile instance variable for actions requiring a specific profile.
    # @return [void]
    @profile = Profile.find(params[:id])
  end

  def profile_params
    # Permits the allowed parameters for profiles.
    # @return [ActionController::Parameters] the permitted parameters.
    params.require(:profile).permit(:name, :date_of_birth, :gender, :city, :country, :nationality, :date_of_disappearance, :city_of_disappearance, :country_of_disappearance, :found, images: [])
  end
end