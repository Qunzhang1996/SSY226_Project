import numpy as np

class TransformationTest:
    def transformation_mp2km(self, state):
        """Transfer from  km state to mass_point state"""
        x_km, y_km, psi, t, v_km = state[0], state[1], state[2], state[3], state[4]
        x = x_km
        y = y_km
        psi = np.arctan2(np.sin(psi), np.cos(psi))
        v = v_km
        v_s=v*np.cos(psi)
        v_n=v*np.sin(psi)
        s=x
        n=y
        return np.array([v_s, s, n, t, v_n])
    
    def transformation_km2mp(self, state):
        """Transfer from mass_point to km state"""
        v_s, s, n, t, v_n = state[0], state[1], state[2], state[3], state[4]
        x_km = s
        y_km = n
        psi = np.arctan2(np.sin(np.arctan2(v_n, v_s)), np.cos(np.arctan2(v_n, v_s)))
        v_km = np.sqrt(v_s**2+v_n**2)
        return np.array([x_km, y_km, psi, t, v_km])
    
    def input_transform(self, u):
        """Transfer from km input to mass_point input, a, delta to ax,ay"""
        a, delta = u[0], u[1]
        ax = a*np.cos(delta)
        ay = a*np.sin(delta)
        print(np.array([ax, ay]))
        return np.array([ax, ay])
    
    def input_transform_inv(self, u):
        """Transfer from mass_point input to km input, ax,ay to a, delta"""
        ax, ay = u[0], u[1]
        a = np.sqrt(ax**2+ay**2)
        delta = np.arctan2(ay, ax)
        print(np.array([a, delta]))
        return np.array([a, delta])
    

    def test_transformation(self):
        # Initial km state
        initial_state_km = np.array([10, 20, np.pi/4, 5, 30])  # Example values
        initial_input_km = np.array([0.06405456, 0.00018099])
        # Convert km to mp and back to km
        state_mp = self.transformation_km2mp(initial_state_km)
        final_state_km = self.transformation_mp2km(state_mp)

        # Convert km input to mp input and back to km input
        input_mp = self.input_transform(initial_input_km)
        final_input_km = self.input_transform_inv(input_mp)

        # Compare the initial and final states
        assert np.allclose(initial_state_km, final_state_km, atol=1e-6), \
            "Transformation mismatch: {} != {}".format(initial_state_km, final_state_km)
        assert np.allclose(initial_input_km, final_input_km, atol=1e-6), \
            "Transformation mismatch: {} != {}".format(initial_input_km, final_input_km)
        print("Test passed: Transformation is correct.")

# Running the test
test = TransformationTest()
test.test_transformation()
