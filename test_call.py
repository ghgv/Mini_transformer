# En el codigo de los transformers, se llama el Multihead el cual ejecuta el forward con argumentos como se se le hubiera llamado
# explicitamente. Esto se debe a que se hace uso del __call__ en un inheritance
class module():
    def __init__(self):
        print("Inside module")
    def forward(self,x):
        print("En el forward",x)
    def __call__(self,x):
        self.forward(x)


class example(module):
    def __init__(self,x):
        super().__init__()
    def forward(self,x):
        print("In the overloaded forward",x)


d= example(5)
a= d(2)