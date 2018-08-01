def assert_col_vectors(p1, p2):
    """Throw assertions if the p1 and p2 are not column vectors with the same
    dimension in the -1 axis.
    """
    assert len(p1.shape) == 2, \
        "p1 %r has shape %r and p2 %r has shape %r\n" % (p1, p1.shape, p2, p2.shape)
    assert len(p2.shape) == 2, \
        "p1 %r has shape %r and p2 %r has shape %r\n" % (p1, p1.shape, p2, p2.shape)
    assert p1.shape[1] == 1,\
        "The first argument %r has shape %r\n" % (p1, p1.shape)
    assert p2.shape[1] == 1,\
        "The second argument %r has shape %r\n" % (p2, p2.shape)


def assert_array_with_wrong_size(a, N, M):
    """Given an array a we want to make sure that it has shape (N, M)
    """
    assert a.shape[0] == N, "Array %r has shape %r \n" % (a, a.shape)
    assert a.shape[1] == M, "Array %r has shape %r \n" % (a, a.shape)


def assert_vector_with_wrong_size(p1, N):
    """Given a vector p1 we want to make sure that it has shape (N, 1)
    """
    assert p1.shape[0] == N, \
        "Vector p1 %r has shape %r, when it should be (%d, 1)" % (p1, p1.shape, N)
    assert p1.shape[1] == 1, \
        "Vector p1 %r has shape %r, when it should be (%d, 1)" % (p1, p1.shape, N)
