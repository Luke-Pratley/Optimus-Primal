class EmptyProx:
    def fun(self, x):
        return 0

    def prox(self, x, tau):
        return x

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x

    beta = 1


class EmptyGrad:
    def fun(self, x):
        return 0

    def grad(self, x):
        return 0

    beta = 1
