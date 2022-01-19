# RandomForest

## Node:
Node è la classe che rappresenta i singoli nodi dell'albero di decisione.

I suoi attributi indicano rispettivamente:
 - `self.predicted_class` la classe (o le classe, se più di una) che predice
 - `self.feature_index` la feature su cui vengono testati i campioni
 - `self.threshold` la soglia utilizzata per classificare i campioni
 - `self.left` e `self.right` rispettivamente i nodi che corrispondono agli elementi minori e maggiori della soglia
 
## DecisionTree:
DecisionTree è la classe che serve a creare e allenare gli alberi di decisione.

Il metodo **`_find_best_split()`** sceglie casualmente una delle feature e poi trova la feature e il valore tali per cui il *gain* è maggiore.

Il metodo **`_grow_tree()`** serve a creare i nodi dell'albero.

Il metodo **`_predict()`** percorre tutto l'albero per classificare l'esempio in input e restituisce la classe predetta.

I metodi **`predict()`** e **`fit()`** servono a chiamare rispettivamente **`_predict()`** e **`_grow_tree()`**.

## RandomForest:
Nel costruttore viene inizializzato il seed per generare risultati ripetibili.

Il metodo **`_sample()`** produce un *bootstrapped dataset*.

Il metodo **`fit()`** crea gli alberi e li aggiunge alla lista **`self.forest`**.

Il metodo **`predict()`** si occupa della classificazione degli elementi di x.


## Test:
Test è una funzione che all'inizio importa un dataset (a seconda del dataset che viene considerato sono necessari dei piccoli aggiustamenti dei parametri o delle operazioni extra, ad esempio cambiare il carattere che fa da separatore o inverire la posizione della colonna con che contiene le classi di appartenenza degli esempi) e poi codifica le variabii non numeriche.

In seguito divide il dataset e poi inizializza i classificatore, li allena e gli fa classificare degli esempi. Alla fine viene calcolata la precisione dei modelli con **`accuracy_score()`**.
Questo viene ripetuto tante volte quanto indicato dal parametro *repetitions*.
