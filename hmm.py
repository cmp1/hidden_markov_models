class HMMVolRegimeRiskManager(object):
  """Applies a Hidden Markov Model to detect hidden volatility regimes.
  
    Args:
      object: A Pandas DataFrame object containing the requisite close prices,
      features etc. Ensure data index is formatted as `YYYY-MM-DD`.
    
    ### Usage:

    The following example demonstrates the BFGS optimizer attempting to find the
    minimum for a simple two dimensional quadratic objective function.
  
  ```python
    # Instantiate class with HLOC dataframe. 
    hmm = HMMVolRegimeRiskManager(hloc_dataframe)
    
    # In absence of expert opinion, estimate the optimum no. of hidden regimes.
    find_n_optimal_states('returns', 5, 'full', show_plot=True)

    # Train Hidden Markov Model.
    hmm.train_hidden_markov_model ('returns', 'GaussianHMM', 'full', 3)

    # Get plot of hidden volatility regimes.
    hmm.plot_in_sample_hidden_states

  ``` 
  """
  def __init__(self, 
               close_prices_df: pd.DataFrame):
    
    self.close_prices_df = close_prices_df
    self.hmm_model = None
    self.hidden_states = None

  def find_n_optimal_states(self,
                            label: str,
                            max_no_components = 3, 
                            covariance_type = 'full',
                            show_plot = bool):
    """Estimate the optimal number of hidden states using a Gaussian Mixture Model.
    
    Args:
      label: String identifying the target series within the OHLC DataFrame used
        to instantiate the HMMVolRegimeRiskManager class.
      max_no_components: Int, the maximum number of suspected Gaussian components 
        i.e. hidden regimes over which to evaluate the Bayesian Information 
        Criterion over.
      covariance_type: String describing the full type covariance parameters to
        use. Must be one of:
          ‘full’: Each component has its own general covariance matrix. 
          ‘tied’: All components share the same general covariance matrix.
          ‘diag’: Each component has its own diagonal covariance matrix.
          ‘spherical’: Each component has its own single variance.
         Default = 'full'.
      show_plot: Bool indicating whether to produce a plot of the BIC w.r.t to
        n_components.
    Returns:
      models: List of dtype int containing the BIC scores w.r.t to number of 
        model components. 

    """
    # Format input sequences.
    X = np.array(self.close_prices_df[label]).reshape(-1, 1)
    # Configure model component criterion.
    n_components = np.arange(1, max_no_components)
    bic = np.zeros(n_components.shape)
    models = []
    # Loop through each number of Gaussians and compute BIC.
    for i, j in enumerate(n_components):
      # Instantiate GMM with j components
      gmm = GaussianMixture(n_components = j,
                            covariance_type = covariance_type)
      gmm.fit(X)
      #compute the BIC for this model
      bic[i] = gmm.bic(X)
      #Add the best-fit model with j components to the list of models
      models.append(gmm)

    if show_plot == True:
      bic_plot = plt.plot(n_components, bic, label = 'BIC')
      plt.hlines(min(bic), min(n_components), max(n_components), color = 'r',
                 linestyle = '--', label = 'Best BIC Score')
      plt.xlabel('n_components')
      plt.ylabel('Bayesian Information Criterion')
      plt.title('BIC Score')
      plt.legend(loc='best')
      return models, bic_plot
    elif show_plot == False:
      return models  
      
  def train_hidden_markov_model(self,
                                label: str, 
                                model_type = 'GaussianHMM',
                                covariance_type = 'full',
                                no_hidden_states = 3):
    """Train Hidden Markov Model.
      Args:
        label: Numpy array, shape (n_samples, n_features). 
          The feature matrix of individual samples.
        model_type: String. Select model of type: 
          GMMGHMM: Conditioned on the state, the observations are modeled as a 
          sample from a finite Gaussian mixture or; 
          GaussianHMM: Conditioned on the state, the observations are modeled as 
          a sample from a single Gaussian distribution.
        covariance_type: String describing the full type covariance parameters to
          use. Must be one of:
            ‘full’: Each component has its own general covariance matrix.
            ‘tied’: All components share the same general covariance matrix.
            ‘diag’: Each component has its own diagonal covariance matrix.
            ‘spherical’: Each component has its own single variance. 
          Default = True.  
        no_of_hidden_states: Int representing the number of hidden states that
          generated the underlying data distribution.    
      Returns:
        close_prices_df: The pd.DataFrame used to instantiate 
          HMMVolRegimeRiskManager class, with the computed hidden states w.r.t 
          the input time series appended.  
    """
    # Reshape inputs.
    X = np.array(self.close_prices_df[label]).reshape(-1, 1)
    
    # GaussianHMM architecture
    if model_type == 'GaussianHMM':    
      hmm_model = GaussianHMM(n_components = no_hidden_states,
                              covariance_type = covariance_type,
                              algorithm = 'viterbi',
                              n_iter = 100).fit(X)
      hidden_states = hmm_model.predict(X)
      self.hmm_model = hmm_model
      self.hidden_states = hidden_states
      variance_by_state = [
        np.diag(hmm_model.covars_[i]) for i in range(hmm_model.n_components)]
      hidden_state_volatilities = {i: v for i, v in enumerate(variance_by_state)}
      no_of_states = [i for i in range(hmm_model.n_components)]
      transition_probabilities = hmm_model.transmat_
      transition_probabilities = pd.DataFrame(transition_probabilities, 
                                              index = no_of_states, 
                                              columns = no_of_states)
      transition_probabilities = np.round(transition_probabilities, 2)

      # Results for stdout
      print(f'Model Convergence: {hmm_model.monitor_.converged}', '\n')
      print('-' * 30)
      print(f'Most Volatile Regime: {max(hidden_state_volatilities, key=hidden_state_volatilities.get)}')
      print(f'Current State: {int(hmm_model.decode(X[-1].reshape(-1, 1))[1])}')
      print(f'Transition Probabilities: \n {transition_probabilities}')
      print('-' * 30)
      print('Mean and Variance of each hidden state:')  
      for i in range(hmm_model.n_components):
        print('Hidden State: {0}'.format(i))
        print(f'Mean = {hmm_model.means_[i]}')
        print(f'Variance = {np.diag(hmm_model.covars_[i])}', '\n')
      print('-' * 30)
    
    # Gaussian Mixture Model HMM architecture, for discrete outcomes. 
    elif model_type == 'GMMHMM':
      hmm_model = GMMHMM(n_components = no_hidden_states,
                         covariance_type = covariance_type,
                         algorithm = 'viterbi',
                         n_iter = 100).fit(X)
      hidden_states = hmm_model.predict(X)
      self.hmm_model = hmm_model
      self.hidden_states = hidden_states
      variance_by_state = [
        np.diag(hmm_model.covars_[i]) for i in range(hmm_model.n_components)]
      hidden_state_volatilities = {i: v for i, v in enumerate(variance_by_state)}
      no_of_states = [i for i in range(hmm_model.n_components)]
      transition_probabilities = hmm_model.transmat_
      transition_probabilities = pd.DataFrame(transition_probabilities, 
                                              index = no_of_states, 
                                              columns = no_of_states)
      transition_probabilities = np.round(transition_probabilities, 2)

      # Results for stdout
      print(f'Model Convergence: {self.hmm_model.monitor_.converged}', '\n')
      print('-' * 30)
      print(f'Most Volatile Regime: {max(hidden_state_volatilities, key=hidden_state_volatilities.get)}')
      print(f'Current State: {int(hmm_model.decode(X[-1].reshape(-1, 1))[1])}')
      print(f'Transition Probabilities: \n {transition_probabilities}')
      print('-' * 30)
      print('Mean and Variance of each hidden state:')  
      for i in range(hmm_model.n_components):
        print('Hidden State: {0}'.format(i))
        print(f'Mean = {hmm_model.means_[i]}')
        print(f'Variance = {np.diag(hmm_model.covars_[i])}', '\n')
      print('-' * 30)

  def create_states_df(self):
    if self.hidden_states is None:
      raise ValueError('Hidden States is NONE: run `train_hidden_markov_model` first')
    else: 
      # Append predicted regimes to original HLOC DataFrame.   
      states_df = pd.DataFrame(self.hidden_states, 
                              columns = ['state'],
                              index = self.close_prices_df.index)
      self.close_prices_df = pd.concat([self.close_prices_df, states_df],
                                        join = 'inner', axis = 1)
    return self.close_prices_df    
      
  def plot_in_sample_hidden_states(self, adjusted_price: str):
    """Plot hidden states for a given curve.
      Args:
        adjusted_price: String referencing the adjusted price 
          curve in the DataFrame object used to instantiate the class.
      Returns:
        plot: A matplotlib time series plot of each detected volatility regime.
    """
    # Chart style configs
    sns.set(font_scale=1.25)
    style_kwds = {'xtick.major.size': 3, 
                  'ytick.major.size': 3,
                  'font.family':u'courier prime code', 
                  'legend.frameon': True,
                  'lineweight': .3}
    sns.set_style('white', style_kwds)
    
    # Plot configs.
    fig, axs = plt.subplots(self.hmm_model.n_components, 
                            sharex=True, 
                            sharey=True, 
                            figsize=(12, 10))
    colors = plt.cm.coolwarm(np.linspace(0, 1, self.hmm_model.n_components))
    for i, (ax, color) in enumerate(zip(axs, colors)):
      mask = self.hidden_states == i
      ax.plot(self.close_prices_df.index.values[mask],
              self.close_prices_df[adjusted_price][mask],
              ".", c=color, ms = 2.5)
      ax.set_title('Hidden State: {0}'.format(i), 
                   fontsize=14, 
                   fontweight='bold')   
      plt.tight_layout()
