Rails.application.routes.draw do
  devise_for :users
  namespace :admin do
    resources :users do
      member do
        get :assign_role
        patch :update_role
        delete :destroy
      end
    end
    resources :profiles
  end
  resources :profiles do
    resources :inquires, only: [:index, :new, :create, :show, :edit, :update, :destroy]
  end
  get "dashboard", to: "dashboards#show"
  root to: "pages#home"
end
