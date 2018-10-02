# arpes.io module

**arpes.io.simple\_load(fragment, df: pandas.core.frame.DataFrame =
None, basic\_prep=True)**

**arpes.io.load\_dataset(dataset\_uuid=None, filename=None, df:
pandas.core.frame.DataFrame = None)**

> You might want to prefer `simple_load` over calling this directly as
> it is more convenient.
> 
>   - Parameters  
>     **dataset\_uuid** – UUID of dataset to load, typically you get
>     this from ds.loc\[‘…’\].id. This actually also
> 
> accepts a dataframe slice so ds.loc\[‘…’\] also works. :param df:
> dataframe to use to lookup the data in. If none is provided, the
> result of default\_dataset is used. :return:

**arpes.io.save\_dataset(arr: Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\], filename=None, force=False)**

> Persists a dataset to disk. In order to serialize some attributes, you
> may need to modify wrap and unwrap arrs above in order to make sure a
> parameter is saved.
> 
> In some cases, such as when you would like to add information to the
> attributes, it is nice to be able to force a write, since a write
> would not take place if the file is already on disk. To do this you
> can set the `force` attribute. :param arr: :param force: :return:

**arpes.io.delete\_dataset(arr\_or\_uuid)**

**arpes.io.dataset\_exists(dataset)**

**arpes.io.is\_a\_dataset(dataset)**

**arpes.io.load\_dataset\_attrs(dataset\_uuid)**

**arpes.io.easy\_pickle(data\_or\_str, name=None)**

**arpes.io.sld(fragment, df: pandas.core.frame.DataFrame = None,
basic\_prep=True)**

**arpes.io.stitch(df\_or\_list, attr\_or\_axis, built\_axis\_name=None,
sort=True)**

> Stitches together a sequence of scans or a DataFrame in order to
> provide a unified dataset along a specified axis
> 
>   - Parameters
>     
>       - **df\_or\_list** – list of the files to load
>     
>       -   - **attr\_or\_axis** – coordinate or attribute in order to  
>             promote to an index. I.e. if ‘t\_a’ is specified,
> 
> we will create a new axis corresponding to the temperature and
> concatenate the data along this axis :return: