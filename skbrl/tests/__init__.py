import pytest
import os

is_skipping_performance_test = os.environ.get('TEST_PERFORMANCE') != "TRUE"
performance_test = pytest.mark.skipif(is_skipping_performance_test,
                                      reason='skip performance tests')
