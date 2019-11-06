from varz.spec import _extract_prefix_and_f


def test_extract_prefix_and_f():
    def f():
        pass

    assert _extract_prefix_and_f(None) == ('', None)
    assert _extract_prefix_and_f(f) == ('', f)
    assert _extract_prefix_and_f('test') == ('test', None)
