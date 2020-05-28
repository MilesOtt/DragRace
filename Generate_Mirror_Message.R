
#I got a lot of inspiration from Jacqueline Nolis' license plate project: 
#https://github.com/jnolis/banned-license-plates/blob/master/train_model.R

#I also learned a lot from Julia Silge's blogpost: 
#https://www.r-bloggers.com/tensorflow-jane-austen-and-text-generation/


#----Loading in Libraries

#install.packages("tensorflow")
library(tensorflow)
install_tensorflow()
library(dplyr)
library(tidyr)
library(tidyverse)
library(stringr)
library(readr)
library(keras)
library(tokenizers)
use_condaenv("r-tensorflow")
#--- Getting the data set up

mirror_data <- 
  read_csv("data/RPDR_Mirror_Messages.csv")%>% #reading the csv
  select(text)%>%                              #just using the mirror text
  str_replace_all("[^[:alnum:] ]", "")%>%         #getting rid of all the other characters
  strsplit(split="") #splitting this up into individual characters
  

mirror_data<-mirror_data[[1]]

print(sprintf("Corpus length: %d", length(mirror_data)))
  
max_length<-20 

chars<-mirror_data %>%
  unique() %>%
  sort()

#cutting the dataset into pieces of max_length text 
#this seems artificial to me since we started with units that made sense..

dataset <- map(
  seq(1, length(mirror_data) - max_length - 1, by = 3), 
  ~list(sentence = mirror_data[.x:(.x + max_length - 1)], 
        next_char = mirror_data[.x + max_length])
)

dataset <- transpose(dataset)

#vectorize

vectorize <- function(data, chars, max_length){
  x <- array(0, dim = c(length(data$sentence), max_length, length(chars)))
  y <- array(0, dim = c(length(data$sentence), length(chars)))
  
  for(i in 1:length(data$sentence)){
    x[i,,] <- sapply(chars, function(x){
      as.integer(x == data$sentence[[i]])
    })
    y[i,] <- as.integer(chars == data$next_char[[i]])
  }
  
  list(y = y,
       x = x)
}

vectors <- vectorize(dataset, chars, max_length)

#model definition

create_model <- function(chars, max_length){
  keras_model_sequential() %>%
    layer_lstm(128, input_shape = c(max_length, length(chars))) %>%
    layer_dense(length(chars)) %>%
    layer_activation("softmax") %>% 
    compile(
      loss = "categorical_crossentropy", 
      optimizer = optimizer_rmsprop(lr = 0.01)
    )
}

#fit the model to a set number of epochs whatever that means

fit_model <- function(model, vectors, epochs = 1){
  model %>% fit(
    vectors$x, vectors$y,
    batch_size = 128,
    epochs = epochs
  )
  NULL
}

#Model Training and results

generate_phrase <- function(model, text, chars, max_length, diversity){
  
  # this function chooses the next character for the phrase
  choose_next_char <- function(preds, chars, temperature){
    preds <- log(preds) / temperature
    exp_preds <- exp(preds)
    preds <- exp_preds / sum(exp(preds))
    
    next_index <- rmultinom(1, 1, preds) %>% 
      as.integer() %>%
      which.max()
    chars[next_index]
  }
  
  # this function takes a sequence of characters and turns it into a numeric array for the model
  convert_sentence_to_data <- function(sentence, chars){
    x <- sapply(chars, function(x){
      as.integer(x == sentence)
    })
    array_reshape(x, c(1, dim(x)))
  }
  
  # the inital sentence is from the text
  start_index <- sample(1:(length(text) - max_length), size = 1)
  sentence <- text[start_index:(start_index + max_length - 1)]
  generated <- ""
  
  # while we still need characters for the phrase
  for(i in 1:(max_length * 20)){
    
    sentence_data <- convert_sentence_to_data(sentence, chars)
    
    # get the predictions for each next character
    preds <- predict(model, sentence_data)
    
    # choose the character
    next_char <- choose_next_char(preds, chars, diversity)
    
    # add it to the text and continue
    generated <- str_c(generated, next_char, collapse = "")
    sentence <- c(sentence[-1], next_char)
  }
  
  generated
}

iterate_model <- function(model, text, chars, max_length, 
                          diversity, vectors, iterations){
  for(iteration in 1:iterations){
    
    message(sprintf("iteration: %02d ---------------\n\n", iteration))
    
    fit_model(model, vectors)
    
    for(diversity in c(0.2, 0.5, 1)){
      
      message(sprintf("diversity: %f ---------------\n\n", diversity))
      
      current_phrase <- 1:10 %>% 
        map_chr(function(x) generate_phrase(model,
                                            text,
                                            chars,
                                            max_length, 
                                            diversity))
      
      message(current_phrase, sep="\n")
      message("\n\n")
      
    }
  }
  NULL
}

#Run the model

model <- create_model(chars, max_length)

iterate_model(model, mirror_data, chars, max_length, diversity, vectors, 40)

