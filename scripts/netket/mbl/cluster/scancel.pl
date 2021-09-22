use File::Copy;
use Data::Dumper;
use Cwd;
$dir = getcwd;

 

for($val = 33179711; ($val <= 33179811); $val+=1)
{
	system "scancel $val";
}