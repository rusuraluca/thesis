class InquiresController < ApplicationController
  # Controller for handling inquire-related actions.

  before_action :set_inquire, only: [:show, :edit, :update, :destroy]
  def index
    # Displays a list of inquiries for a specific profile, paginated.
    # @return [void]
    @profile = Profile.find(params[:profile_id])
    @inquires = @profile.inquires.page(params[:page]).per(3)

  end

  def new
    # Initializes a new inquiry for a specific profile.
    # @return [void]
    @profile = Profile.find(params[:profile_id])
    @inquire = Inquire.new
  end

  def create
    # Creates a new inquiry for a specific profile.
    # @return [void]
    @profile = Profile.find(params[:profile_id])
    @inquire = @profile.inquires.new(inquire_params)
    @inquire.user = current_user

    if @inquire.save
      similarity = fetch_similarity(@profile.images, @inquire.images)
      status = determine_status(similarity)

      if @inquire.update(similarity: similarity, status: status)
        redirect_to [:admin, @profile], notice: 'Your inquiry has been sent successfully and similarity updated.'
      else
        flash.now[:alert] = 'Failed to update similarity: ' + @inquire.errors.full_messages.join(', ')
        render :new, status: :unprocessable_entity
      end
    else
      render :new, status: :unprocessable_entity
    end
  end

  def show
    # Shows the details of a specific inquiry.
  end

  def edit
    # Renders the edit form for a specific inquiry.
  end

  def update
    # Updates a specific inquiry with the provided parameters.
    # @return [void]
    if @inquire.update(inquire_params)
      redirect_to profile_inquire_path(@inquire.profile, @inquire), notice: 'Inquiry was successfully updated.'
    else
      render :edit
    end
  end

  def destroy
    # Deletes a specific inquiry.
    # @return [void]
    @inquire.destroy
    redirect_to profile_inquires_path(@inquire.profile), notice: 'Inquiry was successfully deleted.'
  end

  private

  require 'faraday'
  require 'json'

  def fetch_similarity(profile_images, inquiry_images)
    # Fetches similarity score between profile and inquiry images.
    # @param profile_images [ActiveStorage::Attached::Many] images attached to the profile.
    # @param inquiry_images [ActiveStorage::Attached::Many] images attached to the inquiry.
    # @return [Float] similarity score between the images.
    url = 'https://infinite-pegasus-cheaply.ngrok-free.app/batch_images_similarity'
    conn = Faraday.new(url) do |f|
      f.request :multipart
      f.adapter Faraday.default_adapter
    end

    payload = {
      'imageList1' => convert_to_uploads(profile_images),
      'imageList2' => convert_to_uploads(inquiry_images)
    }

    payload.each do |key, files|
      files.each do |file|
        logger.info "File prepared for upload: #{file.original_filename}"
      end
    end

    response = conn.post do |req|
      req.body = {}
      payload.each do |key, files|
        files.each_with_index do |file, index|
          req.body["#{key}"] = Faraday::FilePart.new(file.path, file.content_type, file.original_filename)
        end
      end
    end

    result = JSON.parse(response.body)
    logger.info "Response received: #{result}"
    result['similarity']
  rescue => e
    logger.error "Failed to call MPAIFR API: #{e.message}"
    0.0
  end

  def convert_to_uploads(attachments)
    # Converts ActiveStorage attachments to Faraday upload objects.
    # @param attachments [ActiveStorage::Attached::Many] images attached to the record.
    # @return [Array<Faraday::UploadIO>] array of Faraday upload objects.
    attachments.map do |attachment|
      file_path = ActiveStorage::Blob.service.path_for(attachment.key)
      logger.info "Accessing file at: #{file_path}"
      if File.exist?(file_path)
        Faraday::UploadIO.new(file_path, attachment.content_type, File.basename(file_path))
      else
        logger.error "File does not exist at path: #{file_path}"
        nil
      end
    end.compact
  end

  def set_inquire
    # Sets the inquiry instance variable for actions requiring a specific inquiry.
    # @return [void]
    @inquire = Inquire.find(params[:id])
  end

  def inquire_params
    # Permits the allowed parameters for inquiries.
    #
    # @return [ActionController::Parameters] the permitted parameters.
    params.require(:inquire).permit(:date_taken, :city_taken, :country_taken, :status, :similarity, images: [])
  end

  def determine_status(similarity)
    # Determines the status of an inquiry based on the similarity score.
    # @param similarity [Float] the similarity score between images.
    # @return [String] the status of the inquiry.
    if similarity >= 0.5
      'Solved'
    elsif similarity > 0.25
      'Not Verified'
    else
      'Not Solved'
    end
  end
end
