% source: http://sachinashanbhag.blogspot.de/2012/09/setting-up-random-number-generator-seed.html

function rng(x)
  if strcmp(x, "default")
    randn("seed")
    rand("seed")
  elseif strcmp(x, "shuffle")
    randn("seed", "reset")
    rand("seed", "reset")
  end
end