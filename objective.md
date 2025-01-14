mainObjective: vectorize a given input

objective Prompt {
  - provide a prompt template that the user can fill values in.
}

objective Vector {
  - hold generated vector
  - hold the input
    - image
    - text
    - audio (future)
    - video (future)
  - be able to
    - get dimensionality
    - get data type of the input
}

objectives DataType {
  Image {
    - use DynamicImage as its backend
  }
  
  Text {
    - use String as its backend
  }
  
  Audio {
    - use ndarray as its backend
  }
  
  Video {
    - use ndarray as its backend
  }
}

objective Vectorize {
  - get an input
  - perform vectorization through
    - LLM
    - Embedding Model (future)
    - Or else (future)
  - return `Vector`
}