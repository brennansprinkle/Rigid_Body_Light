from Rigid import c_rigid_obj as crigid


class RigidBody:
    X_shape = None
    Q_shape = None

    def __init__(self):
        self.cb = crigid.CManyBodies()

    def get_config(self):
        X, Q = self.cb.getConfig()

        X = X.reshape(self.X_shape)
        Q = Q.reshape(self.Q_shape)
        return X, Q

    def set_config(self, X, Q):
        self.__check_and_set_shapes(X, Q)
        X = X.flatten()
        Q = Q.flatten()
        self.cb.setConfig(X, Q)

    def __check_and_set_shapes(self, X, Q):
        self.X_shape = X.shape
        self.Q_shape = Q.shape

        if self.X_shape[0] != self.Q_shape[0]:
            raise RuntimeError("X and Q must have the same number of bodies")
