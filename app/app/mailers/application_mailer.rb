class ApplicationMailer < ActionMailer::Base
  # The base mailer class for the application.
  # All other mailers should inherit from this class.

  # Sets the default "from" email address for all mailers.
  default from: '74f145001@smtp-brevo.com'
  # Specifies the layout to be used for all mailer views.
  layout 'mailer'
end
